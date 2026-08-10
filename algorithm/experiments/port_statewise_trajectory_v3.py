"""Statewise, duration-normalized port trajectory experiment (version 3).

This module is deliberately disjoint from the frozen port experiments.  It
defines a finite action-family contract for one port decision state and adds
the resource semantics that the earlier trajectory experiment did not model:

* tug, navigation-channel, and transfer-team reservations persist until a
  transition completes;
* re-berthing, crane reassignment, yard transfer, and generalized migration
  carry explicit delay, lost-work, and risk costs;
* candidate scores are normalized by their (possibly different) durations;
* only score-semantic dominance valid for every nonnegative queue is pruned
  before the robust score is maximized; terminal-cost Pareto status is audited
  separately and never overrides the robust oracle; and
* public BACASP source-core evidence and synthetic migration evidence are
  emitted as separate, non-substitutable certificates.

The code is a deterministic experiment-side oracle.  It is not connected to a
terminal control plane and does not support an industrial-deployment claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.port_public_benchmark_adapter import (
    build_port_public_benchmark,
)


EPS = 1e-9
SCHEMA_VERSION = "scheduleurm.port.statewise_trajectory.v3"
SPT_SAFETY_POLICY = "spt_statewise_safety"
RECONFIGURATION_SAFETY_POLICY = "reconfiguration_greedy_statewise_safety"
ROBUST_POLICY = "statewise_robust_trajectory_v3"
SAFETY_POLICIES = (SPT_SAFETY_POLICY, RECONFIGURATION_SAFETY_POLICY)
PARETO_COORDINATES = ("makespan", "mean_flow_time", "weighted_tardiness")
TRANSITION_KINDS = {
    "migration",
    "reberth",
    "crane_reassignment",
    "yard_transfer",
}
RESOURCE_KINDS = {"tug", "channel", "transfer"}
REQUIRED_RESOURCE_KINDS = {
    "migration": frozenset({"tug", "channel", "transfer"}),
    "reberth": frozenset({"tug", "channel"}),
    "crane_reassignment": frozenset({"transfer"}),
    "yard_transfer": frozenset({"transfer"}),
}


class PortStatewiseV3Error(RuntimeError):
    """Raised when the finite statewise oracle cannot certify its input."""


def _finite_nonnegative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _finite_positive(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _resource_kind(token: str) -> str:
    if ":" not in token:
        raise ValueError(f"resource token lacks a kind prefix: {token!r}")
    kind, identifier = token.split(":", 1)
    if kind not in RESOURCE_KINDS or not identifier:
        raise ValueError(f"unsupported resource token: {token!r}")
    return kind


@dataclass(frozen=True)
class PortJobState:
    vessel_id: str
    queue_weight: float
    remaining_work: float

    def validate(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id is required")
        _finite_nonnegative(self.queue_weight, f"{self.vessel_id}.queue_weight")
        _finite_positive(self.remaining_work, f"{self.vessel_id}.remaining_work")


@dataclass(frozen=True)
class GeneralizedTransitionCost:
    """Auditable port migration cost in a common penalty unit.

    Delay includes nonproductive transition and warm-up time.  Lost work is
    recorded separately because re-handling or interrupted service need not be
    proportional to elapsed time.  Risk is expected loss, not an unbounded
    qualitative surcharge.
    """

    fixed_cost: float = 0.0
    operation_delay: float = 0.0
    delay_unit_cost: float = 1.0
    lost_work: float = 0.0
    lost_work_unit_cost: float = 1.0
    risk_probability: float = 0.0
    risk_impact: float = 0.0
    resource_cost: float = 0.0

    def validate(self) -> None:
        for label, value in (
            ("fixed_cost", self.fixed_cost),
            ("operation_delay", self.operation_delay),
            ("delay_unit_cost", self.delay_unit_cost),
            ("lost_work", self.lost_work),
            ("lost_work_unit_cost", self.lost_work_unit_cost),
            ("risk_probability", self.risk_probability),
            ("risk_impact", self.risk_impact),
            ("resource_cost", self.resource_cost),
        ):
            _finite_nonnegative(value, label)
        if self.risk_probability > 1.0:
            raise ValueError("risk_probability must not exceed one")

    @property
    def delay_penalty(self) -> float:
        return self.operation_delay * self.delay_unit_cost

    @property
    def lost_work_penalty(self) -> float:
        return self.lost_work * self.lost_work_unit_cost

    @property
    def risk_penalty(self) -> float:
        return self.risk_probability * self.risk_impact

    @property
    def total_penalty(self) -> float:
        self.validate()
        return (
            self.fixed_cost
            + self.delay_penalty
            + self.lost_work_penalty
            + self.risk_penalty
            + self.resource_cost
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "fixed_cost": self.fixed_cost,
            "operation_delay": self.operation_delay,
            "delay_unit_cost": self.delay_unit_cost,
            "delay_penalty": self.delay_penalty,
            "lost_work": self.lost_work,
            "lost_work_unit_cost": self.lost_work_unit_cost,
            "lost_work_penalty": self.lost_work_penalty,
            "risk_probability": self.risk_probability,
            "risk_impact": self.risk_impact,
            "risk_penalty": self.risk_penalty,
            "resource_cost": self.resource_cost,
            "total_penalty": self.total_penalty,
        }


@dataclass(frozen=True)
class PersistentTransition:
    transition_id: str
    kind: str
    vessel_id: str
    start_time: float
    complete_at: float
    resource_tokens: tuple[str, ...]
    cost: GeneralizedTransitionCost = field(default_factory=GeneralizedTransitionCost)

    def validate(self) -> None:
        if not self.transition_id or not self.vessel_id:
            raise ValueError("transition_id and vessel_id are required")
        if self.kind not in TRANSITION_KINDS:
            raise ValueError(f"unsupported transition kind: {self.kind}")
        start = _finite_nonnegative(self.start_time, "transition.start_time")
        complete = _finite_positive(self.complete_at, "transition.complete_at")
        if complete <= start + EPS:
            raise ValueError("transition completion must follow its start")
        if not self.resource_tokens or len(set(self.resource_tokens)) != len(
            self.resource_tokens
        ):
            raise ValueError("transition resources must be nonempty and unique")
        kinds = {_resource_kind(token) for token in self.resource_tokens}
        missing = REQUIRED_RESOURCE_KINDS[self.kind] - kinds
        if missing:
            raise ValueError(
                f"{self.kind} transition lacks persistent resources: {sorted(missing)}"
            )
        self.cost.validate()

    @property
    def duration(self) -> float:
        return self.complete_at - self.start_time

    def snapshot(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "kind": self.kind,
            "vessel_id": self.vessel_id,
            "start_time": self.start_time,
            "complete_at": self.complete_at,
            "duration": self.duration,
            "resource_tokens": list(self.resource_tokens),
            "cost": self.cost.snapshot(),
        }


@dataclass(frozen=True)
class PortDecisionStateV3:
    state_id: str
    clock: float
    jobs: tuple[PortJobState, ...]
    resource_catalog: tuple[str, ...]
    active_transitions: tuple[PersistentTransition, ...] = ()

    def validate(self) -> None:
        if not self.state_id:
            raise ValueError("state_id is required")
        _finite_nonnegative(self.clock, "clock")
        if not self.jobs:
            raise ValueError("at least one unfinished vessel is required")
        for job in self.jobs:
            job.validate()
        job_ids = [job.vessel_id for job in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("vessel identifiers must be unique")
        if not self.resource_catalog or len(set(self.resource_catalog)) != len(
            self.resource_catalog
        ):
            raise ValueError("resource_catalog must be nonempty and unique")
        for token in self.resource_catalog:
            _resource_kind(token)
        transition_ids: set[str] = set()
        for transition in self.active_transitions:
            transition.validate()
            if transition.transition_id in transition_ids:
                raise ValueError("active transition identifiers must be unique")
            transition_ids.add(transition.transition_id)
            if transition.start_time > self.clock + EPS:
                raise ValueError("an active transition cannot start in the future")
            if transition.complete_at <= self.clock + EPS:
                raise ValueError("completed transitions must be removed from the state")
            unknown = set(transition.resource_tokens) - set(self.resource_catalog)
            if unknown:
                raise ValueError(f"active transition uses unknown resources: {sorted(unknown)}")
        _assert_no_transition_conflicts(self.active_transitions)

    @property
    def queue_vector(self) -> dict[str, float]:
        return {job.vessel_id: job.queue_weight for job in self.jobs}

    @property
    def remaining_work(self) -> dict[str, float]:
        return {job.vessel_id: job.remaining_work for job in self.jobs}


@dataclass(frozen=True)
class TrajectoryCandidateV3:
    candidate_id: str
    policy_family: str
    duration: float
    cumulative_lower_service: tuple[tuple[str, float], ...]
    terminal_costs: tuple[tuple[str, float], ...]
    transitions: tuple[PersistentTransition, ...] = ()
    bounded_nontransition_penalty: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.candidate_id or not self.policy_family:
            raise ValueError("candidate_id and policy_family are required")
        _finite_positive(self.duration, f"{self.candidate_id}.duration")
        service = dict(self.cumulative_lower_service)
        if len(service) != len(self.cumulative_lower_service):
            raise ValueError("cumulative lower-service identifiers must be unique")
        if not service:
            raise ValueError("candidate lower-service vector must be nonempty")
        for vessel, value in service.items():
            if not vessel:
                raise ValueError("lower-service vessel identifier is required")
            _finite_nonnegative(value, f"{self.candidate_id}.service[{vessel}]")
        costs = dict(self.terminal_costs)
        if set(costs) != set(PARETO_COORDINATES):
            raise ValueError("candidate must define the three registered Pareto coordinates")
        for name, value in costs.items():
            _finite_nonnegative(value, f"{self.candidate_id}.{name}")
        _finite_nonnegative(
            self.bounded_nontransition_penalty,
            f"{self.candidate_id}.bounded_nontransition_penalty",
        )
        for transition in self.transitions:
            transition.validate()
        if any(
            transition.complete_at > self.duration + EPS
            or transition.start_time < -EPS
            for transition in self.transitions
        ):
            raise ValueError("candidate transition must lie inside its relative duration")
        _assert_no_transition_conflicts(self.transitions)

    @property
    def service(self) -> dict[str, float]:
        return dict(self.cumulative_lower_service)

    @property
    def costs(self) -> dict[str, float]:
        return dict(self.terminal_costs)

    @property
    def transition_penalty(self) -> float:
        return sum(row.cost.total_penalty for row in self.transitions)

    @property
    def total_penalty(self) -> float:
        return self.bounded_nontransition_penalty + self.transition_penalty

    def duration_normalized_score(self, state: PortDecisionStateV3) -> float:
        queue = state.queue_vector
        numerator = sum(
            queue.get(vessel_id, 0.0) * lower_service
            for vessel_id, lower_service in self.cumulative_lower_service
        ) - self.total_penalty
        score = numerator / self.duration
        if not math.isfinite(score):
            raise ValueError("duration-normalized candidate score is non-finite")
        return score

    def snapshot(self, state: PortDecisionStateV3) -> dict[str, Any]:
        numerator = self.duration_normalized_score(state) * self.duration
        return {
            "candidate_id": self.candidate_id,
            "policy_family": self.policy_family,
            "duration": self.duration,
            "cumulative_lower_service": dict(self.cumulative_lower_service),
            "bounded_nontransition_penalty": self.bounded_nontransition_penalty,
            "transition_penalty": self.transition_penalty,
            "total_penalty": self.total_penalty,
            "unnormalized_robust_numerator": numerator,
            "duration_normalized_score": self.duration_normalized_score(state),
            "terminal_costs": dict(self.terminal_costs),
            "transitions": [row.snapshot() for row in self.transitions],
            "metadata": _canonical(self.metadata),
        }


@dataclass(frozen=True)
class CandidateAssessmentV3:
    candidate: TrajectoryCandidateV3
    statewise_feasible: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class StatewiseSelectionV3:
    state: PortDecisionStateV3
    assessments: tuple[CandidateAssessmentV3, ...]
    score_semantic_frontier_ids: tuple[str, ...]
    terminal_pareto_frontier_ids: tuple[str, ...]
    selected: TrajectoryCandidateV3
    selected_score: float
    oracle_score: float

    @property
    def oracle_gap(self) -> float:
        return max(0.0, self.oracle_score - self.selected_score)

    @property
    def feasible_candidates(self) -> tuple[TrajectoryCandidateV3, ...]:
        return tuple(row.candidate for row in self.assessments if row.statewise_feasible)

    def snapshot(self) -> dict[str, Any]:
        feasible_ids = [row.candidate.candidate_id for row in self.assessments if row.statewise_feasible]
        rejected = {
            row.candidate.candidate_id: list(row.rejection_reasons)
            for row in self.assessments
            if not row.statewise_feasible
        }
        return {
            "state_id": self.state.state_id,
            "finite_candidate_count": len(self.assessments),
            "feasible_candidate_ids": feasible_ids,
            "rejected_candidates": rejected,
            "safety_policy_families": list(SAFETY_POLICIES),
            "score_semantic_frontier_ids": list(self.score_semantic_frontier_ids),
            "score_semantic_pruning_exact": (
                self.selected.candidate_id in self.score_semantic_frontier_ids
                and self.oracle_gap <= EPS
            ),
            "terminal_pareto_frontier_ids": list(self.terminal_pareto_frontier_ids),
            "selected_terminal_pareto": (
                self.selected.candidate_id in self.terminal_pareto_frontier_ids
            ),
            "selected_candidate": self.selected.snapshot(self.state),
            "oracle_duration_normalized_score": self.oracle_score,
            "oracle_gap_alpha0": self.oracle_gap,
            "oracle_gap_alpha1": 0.0,
            "score_semantics": (
                "(queue_weight dot cumulative_lower_service - generalized_penalty) / duration"
            ),
            "persistent_transition_resources_checked": True,
            "finite_family_penalty_bound_P0": max(
                row.candidate.total_penalty
                for row in self.assessments
                if row.statewise_feasible
            ),
        }


def _overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> bool:
    return max(left_start, right_start) < min(left_end, right_end) - EPS


def _assert_no_transition_conflicts(transitions: Sequence[PersistentTransition]) -> None:
    for index, left in enumerate(transitions):
        for right in transitions[index + 1 :]:
            if not _overlap(left.start_time, left.complete_at, right.start_time, right.complete_at):
                continue
            shared = set(left.resource_tokens) & set(right.resource_tokens)
            if shared:
                raise ValueError(
                    "overlapping transitions share persistent resources: "
                    f"{sorted(shared)}"
                )
            if left.vessel_id == right.vessel_id:
                raise ValueError("one vessel cannot execute overlapping transitions")


def _candidate_rejection_reasons(
    state: PortDecisionStateV3,
    candidate: TrajectoryCandidateV3,
) -> tuple[str, ...]:
    reasons: list[str] = []
    try:
        candidate.validate()
    except (TypeError, ValueError) as exc:
        return (f"candidate_validation:{exc}",)

    jobs = state.remaining_work
    for vessel_id, lower_service in candidate.cumulative_lower_service:
        if vessel_id not in jobs:
            reasons.append(f"unknown_vessel:{vessel_id}")
        elif lower_service > jobs[vessel_id] + EPS:
            reasons.append(f"service_exceeds_remaining_work:{vessel_id}")

    catalog = set(state.resource_catalog)
    absolute_transitions: list[PersistentTransition] = []
    for transition in candidate.transitions:
        unknown = set(transition.resource_tokens) - catalog
        if unknown:
            reasons.append(f"unknown_resources:{','.join(sorted(unknown))}")
        absolute_transitions.append(
            PersistentTransition(
                transition_id=transition.transition_id,
                kind=transition.kind,
                vessel_id=transition.vessel_id,
                start_time=state.clock + transition.start_time,
                complete_at=state.clock + transition.complete_at,
                resource_tokens=transition.resource_tokens,
                cost=transition.cost,
            )
        )

    for transition in absolute_transitions:
        for active in state.active_transitions:
            if not _overlap(
                transition.start_time,
                transition.complete_at,
                active.start_time,
                active.complete_at,
            ):
                continue
            shared = set(transition.resource_tokens) & set(active.resource_tokens)
            if shared:
                reasons.append(
                    "persistent_resource_busy:"
                    + ",".join(sorted(shared))
                    + f":until={active.complete_at:g}"
                )
            if transition.vessel_id == active.vessel_id:
                reasons.append(f"vessel_transition_in_progress:{transition.vessel_id}")
    return tuple(sorted(set(reasons)))


def _terminal_cost_dominates(
    left: TrajectoryCandidateV3, right: TrajectoryCandidateV3
) -> bool:
    left_costs = left.costs
    right_costs = right.costs
    no_worse = all(left_costs[name] <= right_costs[name] + EPS for name in PARETO_COORDINATES)
    strictly_better = any(left_costs[name] < right_costs[name] - EPS for name in PARETO_COORDINATES)
    return no_worse and strictly_better


def _score_semantic_dominates(
    left: TrajectoryCandidateV3,
    right: TrajectoryCandidateV3,
    state: PortDecisionStateV3,
) -> bool:
    """Return dominance that preserves robust score for every nonnegative Q."""

    left_service = left.service
    right_service = right.service
    no_worse_service = all(
        left_service.get(job.vessel_id, 0.0) / left.duration
        >= right_service.get(job.vessel_id, 0.0) / right.duration - EPS
        for job in state.jobs
    )
    no_worse_penalty = (
        left.total_penalty / left.duration
        <= right.total_penalty / right.duration + EPS
    )
    strictly_better = any(
        left_service.get(job.vessel_id, 0.0) / left.duration
        > right_service.get(job.vessel_id, 0.0) / right.duration + EPS
        for job in state.jobs
    ) or (
        left.total_penalty / left.duration
        < right.total_penalty / right.duration - EPS
    )
    return no_worse_service and no_worse_penalty and strictly_better


def select_statewise_trajectory_v3(
    state: PortDecisionStateV3,
    candidates: Iterable[TrajectoryCandidateV3],
    *,
    max_candidates: int = 4096,
) -> StatewiseSelectionV3:
    """Select the exact duration-normalized oracle after theorem-safe pruning."""

    try:
        state.validate()
    except (TypeError, ValueError) as exc:
        raise PortStatewiseV3Error(f"invalid state: {exc}") from exc
    rows = tuple(candidates)
    if not rows or len(rows) > max_candidates:
        raise PortStatewiseV3Error("candidate family must be finite, nonempty, and bounded")
    ids = [row.candidate_id for row in rows]
    if len(ids) != len(set(ids)):
        raise PortStatewiseV3Error("candidate identifiers must be unique")

    assessments = tuple(
        CandidateAssessmentV3(
            candidate=row,
            statewise_feasible=not (reasons := _candidate_rejection_reasons(state, row)),
            rejection_reasons=reasons,
        )
        for row in rows
    )
    feasible = tuple(row.candidate for row in assessments if row.statewise_feasible)
    feasible_policies = {row.policy_family for row in feasible}
    missing_safety = set(SAFETY_POLICIES) - feasible_policies
    if missing_safety:
        raise PortStatewiseV3Error(
            "statewise family lacks feasible safety candidates: "
            + ", ".join(sorted(missing_safety))
        )

    score_semantic_frontier = tuple(
        row
        for row in feasible
        if not any(
            _score_semantic_dominates(other, row, state)
            for other in feasible
            if other is not row
        )
    )
    if not score_semantic_frontier:
        raise PortStatewiseV3Error("score-semantic pruning removed the entire family")
    terminal_frontier = tuple(
        row
        for row in feasible
        if not any(
            _terminal_cost_dominates(other, row)
            for other in feasible
            if other is not row
        )
    )
    all_scored = tuple(
        (row.duration_normalized_score(state), row) for row in feasible
    )
    frontier_scored = tuple(
        (row.duration_normalized_score(state), row)
        for row in score_semantic_frontier
    )
    oracle_score = max(score for score, _ in all_scored)
    if abs(max(score for score, _ in frontier_scored) - oracle_score) > EPS:
        raise PortStatewiseV3Error("score-semantic pruning changed the robust oracle")
    selected = min(
        (
            row
            for score, row in frontier_scored
            if abs(score - oracle_score) <= EPS
        ),
        key=lambda row: row.candidate_id,
    )
    return StatewiseSelectionV3(
        state=state,
        assessments=assessments,
        score_semantic_frontier_ids=tuple(
            sorted(row.candidate_id for row in score_semantic_frontier)
        ),
        terminal_pareto_frontier_ids=tuple(
            sorted(row.candidate_id for row in terminal_frontier)
        ),
        selected=selected,
        selected_score=selected.duration_normalized_score(state),
        oracle_score=oracle_score,
    )


def build_source_core_bacasp_certificate(
    public_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract BACASP source-core evidence without importing migration claims."""

    required = (
        "artifact_hash",
        "source_authenticity_ready",
        "resource_feasibility_ready",
        "source_time_invariant_assignment_ready",
        "mid_service_reconfiguration_admitted",
        "source_core_metrics",
        "source_core_pareto",
    )
    missing = [key for key in required if key not in public_report]
    if missing:
        raise PortStatewiseV3Error(f"public source-core report is incomplete: {missing}")
    ready = all(
        (
            bool(public_report.get("pass")),
            bool(public_report["source_authenticity_ready"]),
            bool(public_report["resource_feasibility_ready"]),
            bool(public_report["source_time_invariant_assignment_ready"]),
            public_report["mid_service_reconfiguration_admitted"] is False,
        )
    )
    if not ready:
        raise PortStatewiseV3Error("BACASP source-core certificate failed closed")
    artifact_hash = str(public_report["artifact_hash"])
    if len(artifact_hash) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_hash.lower()
    ):
        raise PortStatewiseV3Error("BACASP source artifact hash is not SHA-256")
    return {
        "certificate_type": "bacasp_source_core_only",
        "ready": True,
        "source_artifact_hash": artifact_hash,
        "source_core_metrics": _canonical(public_report["source_core_metrics"]),
        "source_core_pareto": _canonical(public_report["source_core_pareto"]),
        "time_invariant_assignment_only": True,
        "mid_service_migration_evidence": False,
        "synthetic_migration_evidence_used": False,
        "scope_boundary": (
            "The certificate covers the pinned BACASP source-core berth/quay mapping only; "
            "it does not validate re-berthing, crane reassignment, yard transfer, or migration."
        ),
    }


def build_synthetic_migration_certificate(
    selection: StatewiseSelectionV3,
) -> dict[str, Any]:
    """Build a separate certificate for synthetic transition semantics."""

    snapshot = selection.snapshot()
    transition_rows = [
        transition.snapshot()
        for assessment in selection.assessments
        if assessment.statewise_feasible
        for transition in assessment.candidate.transitions
    ]
    kinds = sorted({row["kind"] for row in transition_rows})
    cost_components = {
        "delay": all("delay_penalty" in row["cost"] for row in transition_rows),
        "lost_work": all("lost_work_penalty" in row["cost"] for row in transition_rows),
        "risk": all("risk_penalty" in row["cost"] for row in transition_rows),
    }
    ready = bool(
        transition_rows
        and all(cost_components.values())
        and snapshot["score_semantic_pruning_exact"]
        and snapshot["oracle_gap_alpha0"] <= EPS
    )
    return {
        "certificate_type": "synthetic_statewise_migration_only",
        "ready": ready,
        "selection": snapshot,
        "transition_kinds_exercised": kinds,
        "persistent_resource_kinds": sorted(
            {
                _resource_kind(token)
                for row in transition_rows
                for token in row["resource_tokens"]
            }
        ),
        "generalized_cost_components_ready": cost_components,
        "bacasp_source_core_evidence_used": False,
        "physical_terminal_observations_used": False,
        "scope_boundary": (
            "This is a deterministic synthetic semi-Markov transition certificate, "
            "not evidence of execution by an industrial terminal control plane."
        ),
    }


def _demo_state_and_candidates() -> tuple[PortDecisionStateV3, tuple[TrajectoryCandidateV3, ...]]:
    state = PortDecisionStateV3(
        state_id="synthetic_port_state_v3_demo",
        clock=10.0,
        jobs=(
            PortJobState("v_short", queue_weight=4.0, remaining_work=20.0),
            PortJobState("v_long", queue_weight=7.0, remaining_work=60.0),
        ),
        resource_catalog=(
            "tug:t1",
            "tug:t2",
            "channel:main",
            "transfer:team1",
            "transfer:team2",
        ),
        active_transitions=(
            PersistentTransition(
                transition_id="active_reberth",
                kind="reberth",
                vessel_id="v_active",
                start_time=8.0,
                complete_at=14.0,
                resource_tokens=("tug:t1", "channel:main"),
                cost=GeneralizedTransitionCost(operation_delay=6.0),
            ),
        ),
    )
    spt = TrajectoryCandidateV3(
        candidate_id="spt_safe",
        policy_family=SPT_SAFETY_POLICY,
        duration=8.0,
        cumulative_lower_service=(("v_short", 8.0),),
        terminal_costs=(("makespan", 80.0), ("mean_flow_time", 35.0), ("weighted_tardiness", 22.0)),
    )
    reconfiguration_safe = TrajectoryCandidateV3(
        candidate_id="reconfiguration_safe",
        policy_family=RECONFIGURATION_SAFETY_POLICY,
        duration=8.0,
        cumulative_lower_service=(("v_long", 9.0),),
        terminal_costs=(("makespan", 74.0), ("mean_flow_time", 38.0), ("weighted_tardiness", 20.0)),
        transitions=(
            PersistentTransition(
                transition_id="safe_crane_move",
                kind="crane_reassignment",
                vessel_id="v_long",
                start_time=0.0,
                complete_at=2.0,
                resource_tokens=("transfer:team1",),
                cost=GeneralizedTransitionCost(
                    fixed_cost=0.5,
                    operation_delay=2.0,
                    delay_unit_cost=0.2,
                    lost_work=0.2,
                    lost_work_unit_cost=0.5,
                    risk_probability=0.02,
                    risk_impact=2.0,
                ),
            ),
        ),
    )
    robust = TrajectoryCandidateV3(
        candidate_id="robust_delayed_reberth",
        policy_family=ROBUST_POLICY,
        duration=12.0,
        cumulative_lower_service=(("v_short", 7.0), ("v_long", 16.0)),
        terminal_costs=(("makespan", 70.0), ("mean_flow_time", 32.0), ("weighted_tardiness", 18.0)),
        transitions=(
            PersistentTransition(
                transition_id="delayed_reberth",
                kind="reberth",
                vessel_id="v_long",
                start_time=4.0,
                complete_at=9.0,
                resource_tokens=("tug:t2", "channel:main"),
                cost=GeneralizedTransitionCost(
                    fixed_cost=1.0,
                    operation_delay=5.0,
                    delay_unit_cost=0.2,
                    lost_work=0.5,
                    lost_work_unit_cost=0.5,
                    risk_probability=0.03,
                    risk_impact=3.0,
                    resource_cost=0.2,
                ),
            ),
        ),
    )
    conflicting = TrajectoryCandidateV3(
        candidate_id="conflicting_immediate_reberth",
        policy_family=ROBUST_POLICY,
        duration=8.0,
        cumulative_lower_service=(("v_long", 18.0),),
        terminal_costs=(("makespan", 60.0), ("mean_flow_time", 30.0), ("weighted_tardiness", 15.0)),
        transitions=(
            PersistentTransition(
                transition_id="conflicting_reberth",
                kind="reberth",
                vessel_id="v_long",
                start_time=0.0,
                complete_at=3.0,
                resource_tokens=("tug:t1", "channel:main"),
                cost=GeneralizedTransitionCost(operation_delay=3.0),
            ),
        ),
    )
    return state, (spt, reconfiguration_safe, robust, conflicting)


def build_port_statewise_trajectory_v3(
    public_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the disjoint source-core and synthetic-transition v3 report."""

    source_report = (
        dict(public_report)
        if public_report is not None
        else build_port_public_benchmark(determinism_repeats=2)
    )
    source_certificate = build_source_core_bacasp_certificate(source_report)
    state, candidates = _demo_state_and_candidates()
    selection = select_statewise_trajectory_v3(state, candidates)
    migration_certificate = build_synthetic_migration_certificate(selection)
    separated = bool(
        source_certificate["synthetic_migration_evidence_used"] is False
        and migration_certificate["bacasp_source_core_evidence_used"] is False
    )
    ready = bool(source_certificate["ready"] and migration_certificate["ready"] and separated)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "port_statewise_trajectory_v3",
        "pass": ready,
        "status": "PORT_STATEWISE_TRAJECTORY_V3_PASS" if ready else "PORT_STATEWISE_TRAJECTORY_V3_OPEN",
        "finite_statewise_candidate_family_ready": True,
        "duration_normalized_oracle_ready": selection.oracle_gap <= EPS,
        "score_semantic_pruning_ready": (
            selection.selected.candidate_id in selection.score_semantic_frontier_ids
            and selection.oracle_gap <= EPS
        ),
        "terminal_pareto_audit_ready": (
            selection.selected.candidate_id in selection.terminal_pareto_frontier_ids
        ),
        "persistent_tug_channel_transfer_constraints_ready": (
            migration_certificate["persistent_resource_kinds"]
            == ["channel", "transfer", "tug"]
        ),
        "certificate_non_substitution_ready": separated,
        "certificates": {
            "bacasp_source_core": source_certificate,
            "synthetic_migration": migration_certificate,
        },
        "industrial_deployment_claim_ready": False,
        "global_port_optimality_claim_ready": False,
        "claim_boundary": {
            "supports": [
                "a finite statewise candidate family containing both registered safety policies",
                "persistent tug/channel/transfer exclusion over complete transition intervals",
                "duration-normalized robust scoring with score-semantic dominance pruning",
                "a separate terminal-cost Pareto audit that cannot override the robust oracle",
                "separate BACASP source-core and synthetic migration certificates",
            ],
            "does_not_support": [
                "deployment in an operating industrial terminal",
                "safety certification of physical tug, channel, crane, or yard operations",
                "global optimality for arbitrary berth-allocation or crane-scheduling instances",
                "using synthetic migration evidence as BACASP source evidence or vice versa",
            ],
        },
    }
    report["artifact_hash"] = _digest(report)
    return report


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value cannot enter a certificate")
        return round(value, 12)
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "GeneralizedTransitionCost",
    "PersistentTransition",
    "PortDecisionStateV3",
    "PortJobState",
    "PortStatewiseV3Error",
    "RECONFIGURATION_SAFETY_POLICY",
    "ROBUST_POLICY",
    "SPT_SAFETY_POLICY",
    "StatewiseSelectionV3",
    "TrajectoryCandidateV3",
    "build_port_statewise_trajectory_v3",
    "build_source_core_bacasp_certificate",
    "build_synthetic_migration_certificate",
    "select_statewise_trajectory_v3",
]
