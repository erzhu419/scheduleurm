"""Theorem-facing, event-driven heterogeneous port scheduling benchmark.

This sidecar instantiates the Scheduleurm finite configuration-action contract
without touching the live scheduler.  Every decision is made over an explicitly
enumerated, state-feasible family.  The robust policy maximizes

    Q^T lower_service(configuration) - bounded_penalty(configuration)

and records an exact finite-family oracle audit.  The benchmark is intentionally
scoped to registered synthetic instances; it is not evidence from a physical
terminal and it is not a claim of global port-scheduling optimality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.port_scheduling_instances import (
    BerthSpec,
    PortInstance,
    QuayCraneSpec,
    VesselSpec,
    YardBlockSpec,
    built_in_port_instances,
    load_port_instance,
)


EPS = 1e-9
SCHEMA_VERSION = "scheduleurm.port.event_driven.v1"
THEOREM_POLICY = "robust_maxweight"
BASELINE_POLICIES = (
    "fcfs_static",
    "spt_static",
    "edd_static",
    "reconfiguration_greedy",
)
ALL_POLICIES = (THEOREM_POLICY,) + BASELINE_POLICIES
RECONFIGURATION_ACTIONS = {"reberth", "crane_reassignment", "yard_rehandle"}
PRIMARY_COST_METRICS = ("makespan", "mean_flow_time", "weighted_tardiness")


class PortSimulationError(RuntimeError):
    """Raised when a registered instance cannot make a feasible event transition."""


@dataclass(frozen=True)
class AtomicAction:
    action_id: str
    action_type: str
    vessel_id: str
    stage: str
    conflict_tokens: tuple[str, ...]
    lower_service_delta: tuple[tuple[str, float], ...]
    penalty_units: float
    nominal_rate: float
    lower_rate: float
    estimated_duration: float
    estimated_gain_s: float = 0.0
    fixed_cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def theorem_ready(self) -> bool:
        return (
            bool(self.action_id)
            and bool(self.conflict_tokens)
            and self.lower_rate > 0.0
            and self.penalty_units >= 0.0
            and all(math.isfinite(value) for _, value in self.lower_service_delta)
        )

    def delta(self) -> dict[str, float]:
        return dict(self.lower_service_delta)

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "task_id": self.vessel_id,
            "stage": self.stage,
            "resource_ids": list(self.conflict_tokens),
            "lower_service_delta": dict(self.lower_service_delta),
            "selected_class_lower_service": _rounded(self.lower_rate),
            "nominal_rate": _rounded(self.nominal_rate),
            "penalty_units": _rounded(self.penalty_units),
            "estimated_duration": _rounded(self.estimated_duration),
            "estimated_gain_s": _rounded(self.estimated_gain_s),
            "fixed_cost": _rounded(self.fixed_cost),
            "score_semantics": "robust_maxweight_lower_service",
            "theorem_ready": self.theorem_ready,
            "statewise_feasible": True,
            "metadata": _canonical(self.metadata),
        }


@dataclass
class VesselRuntime:
    spec: VesselSpec
    stage: str = "not_arrived"
    remaining_quay: float = 0.0
    remaining_yard: float = 0.0
    remaining_gate: float = 0.0
    berth_id: str | None = None
    crane_ids: list[str] = field(default_factory=list)
    yard_block_id: str | None = None
    gate_lane_id: str | None = None
    quay_start: float | None = None
    yard_start: float | None = None
    gate_start: float | None = None
    completion_time: float | None = None
    reberth_count: int = 0
    crane_reassignment_count: int = 0
    yard_rehandle_count: int = 0

    def __post_init__(self) -> None:
        self.remaining_quay = float(self.spec.quay_work)
        self.remaining_yard = float(self.spec.yard_work)
        self.remaining_gate = float(self.spec.gate_work)


@dataclass(frozen=True)
class PendingTransition:
    kind: str
    vessel_id: str
    complete_at: float
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ConfigurationSelection:
    actions: tuple[AtomicAction, ...]
    objective: float
    robust_score: float
    oracle_robust_score: float
    candidate_configuration_count: int
    lower_service: Mapping[str, float]
    penalty_units: float

    @property
    def oracle_gap_alpha0(self) -> float:
        return max(0.0, self.oracle_robust_score - self.robust_score)


class PortSchedulingSimulator:
    """Finite event-driven simulator with explicit terminal resources."""

    def __init__(self, instance: PortInstance, policy: str = THEOREM_POLICY):
        instance.validate()
        if policy not in ALL_POLICIES:
            raise ValueError(f"unknown port policy: {policy}")
        self.instance = instance
        self.policy = policy
        self.clock = 0.0
        self.next_decision_time = 0.0
        self.vessels = {row.vessel_id: VesselRuntime(row) for row in instance.vessels}
        self.berths = {row.berth_id: row for row in instance.berths}
        self.cranes = {row.crane_id: row for row in instance.quay_cranes}
        self.yards = {row.block_id: row for row in instance.yard_blocks}
        self.gates = {row.lane_id: row for row in instance.gate_lanes}
        self.berth_occupancy: dict[str, str | None] = {key: None for key in self.berths}
        self.crane_assignment: dict[str, str | None] = {key: None for key in self.cranes}
        self.yard_occupancy: dict[str, str | None] = {key: None for key in self.yards}
        self.gate_occupancy: dict[str, str | None] = {key: None for key in self.gates}
        self.transfer_occupancy: str | None = None
        self.transitions: list[PendingTransition] = []
        self.resource_busy_time: dict[str, float] = {
            **{f"berth:{key}": 0.0 for key in self.berths},
            **{f"crane:{key}": 0.0 for key in self.cranes},
            **{f"yard:{key}": 0.0 for key in self.yards},
            **{f"gate:{key}": 0.0 for key in self.gates},
            "yard_transfer": 0.0,
        }
        self.event_trace: list[dict[str, Any]] = []
        self.decision_audits: list[dict[str, Any]] = []
        self.action_counts: Counter[str] = Counter()
        self.reconfiguration_duration = 0.0
        self.reconfiguration_fixed_cost = 0.0
        self.feasibility_checks = 0
        self.feasibility_violations: list[str] = []
        self._event_count = 0
        self._peak_active_vessels = 0

    def run(self, *, max_events: int = 20_000) -> dict[str, Any]:
        self._release_arrivals()
        while not self._all_completed():
            self._dispatch_at_current_time()
            next_time = self._next_event_time()
            if not math.isfinite(next_time):
                raise PortSimulationError(self._deadlock_message("no finite next event"))
            if next_time <= self.clock + EPS:
                raise PortSimulationError(self._deadlock_message("event clock did not advance"))
            self._advance_to(next_time)
            self._process_events_at_clock()
            self._event_count += 1
            if self._event_count > max_events:
                raise PortSimulationError(self._deadlock_message("event limit exceeded"))
        return self._result()

    def feasible_atomic_actions(self) -> tuple[AtomicAction, ...]:
        actions: list[AtomicAction] = []
        actions.extend(self._quay_start_actions())
        actions.extend(self._yard_start_actions())
        actions.extend(self._gate_start_actions())
        actions.extend(self._reberth_actions())
        actions.extend(self._crane_reassignment_actions())
        actions.extend(self._yard_rehandle_actions())
        unique = {action.action_id: action for action in actions}
        out = tuple(unique[key] for key in sorted(unique))
        for action in out:
            ok, reason = self.check_action_feasible(action)
            self.feasibility_checks += 1
            if not ok:
                self.feasibility_violations.append(f"{action.action_id}:{reason}")
                raise PortSimulationError(f"candidate generator emitted infeasible action: {reason}")
            if not action.theorem_ready:
                raise PortSimulationError(f"candidate action lacks theorem-facing fields: {action.action_id}")
        return out

    def check_action_feasible(self, action: AtomicAction) -> tuple[bool, str]:
        runtime = self.vessels.get(action.vessel_id)
        if runtime is None:
            return False, "unknown_vessel"
        meta = action.metadata
        if action.action_type == "start_quay":
            berth = str(meta.get("berth_id") or "")
            cranes = tuple(str(item) for item in meta.get("crane_ids", ()))
            if runtime.stage != "waiting_quay":
                return False, "vessel_not_waiting_quay"
            if self.berth_occupancy.get(berth) is not None:
                return False, "berth_not_idle"
            if not cranes or any(self.crane_assignment.get(crane) is not None for crane in cranes):
                return False, "crane_not_idle"
            if not self._vessel_fits_berth(runtime.spec, self.berths[berth]):
                return False, "berth_incompatible"
            if any(berth not in self.cranes[crane].compatible_berths for crane in cranes):
                return False, "crane_berth_incompatible"
            return True, ""
        if action.action_type == "start_yard":
            block = str(meta.get("yard_block_id") or "")
            if runtime.stage != "waiting_yard":
                return False, "vessel_not_waiting_yard"
            if self.yard_occupancy.get(block) is not None:
                return False, "yard_not_idle"
            if self.yards[block].rate(runtime.spec.cargo_class) <= 0.0:
                return False, "yard_incompatible"
            return True, ""
        if action.action_type == "start_gate":
            lane = str(meta.get("gate_lane_id") or "")
            if runtime.stage != "waiting_gate":
                return False, "vessel_not_waiting_gate"
            if self.gate_occupancy.get(lane) is not None:
                return False, "gate_not_idle"
            if self.gates[lane].rate(runtime.spec.cargo_class) <= 0.0:
                return False, "gate_incompatible"
            return True, ""
        if action.action_type == "reberth":
            source = str(meta.get("source_berth") or "")
            target = str(meta.get("target_berth") or "")
            cranes = tuple(str(item) for item in meta.get("target_cranes", ()))
            if runtime.stage != "quay" or runtime.berth_id != source:
                return False, "vessel_not_at_source_berth"
            if self.berth_occupancy.get(target) is not None:
                return False, "target_berth_not_idle"
            if runtime.reberth_count >= self.instance.max_reberths_per_vessel:
                return False, "reberth_limit"
            allowed_cranes = set(runtime.crane_ids) | {
                key for key, value in self.crane_assignment.items() if value is None
            }
            if not cranes or not set(cranes) <= allowed_cranes:
                return False, "target_crane_unavailable"
            if any(target not in self.cranes[crane].compatible_berths for crane in cranes):
                return False, "target_crane_incompatible"
            return True, ""
        if action.action_type == "crane_reassignment":
            target = runtime
            crane = str(meta.get("crane_id") or "")
            source_id = str(meta.get("source_vessel") or "")
            if target.stage != "quay" or len(target.crane_ids) >= target.spec.max_cranes:
                return False, "target_not_reassignment_ready"
            if target.crane_reassignment_count >= self.instance.max_crane_reassignments_per_vessel:
                return False, "crane_reassignment_limit"
            if source_id:
                source = self.vessels.get(source_id)
                if source is None or source.stage != "quay" or crane not in source.crane_ids or len(source.crane_ids) < 2:
                    return False, "source_cannot_release_crane"
            elif self.crane_assignment.get(crane) is not None:
                return False, "idle_pool_crane_not_idle"
            if target.berth_id not in self.cranes[crane].compatible_berths:
                return False, "crane_target_incompatible"
            return True, ""
        if action.action_type == "yard_rehandle":
            source = str(meta.get("source_yard") or "")
            target = str(meta.get("target_yard") or "")
            if runtime.stage != "yard" or runtime.yard_block_id != source:
                return False, "vessel_not_at_source_yard"
            if self.yard_occupancy.get(target) is not None or self.transfer_occupancy is not None:
                return False, "rehandle_resource_not_idle"
            if runtime.yard_rehandle_count >= self.instance.max_yard_rehandles_per_vessel:
                return False, "yard_rehandle_limit"
            if self.yards[target].rate(runtime.spec.cargo_class) <= 0.0:
                return False, "target_yard_incompatible"
            return True, ""
        return False, "unknown_action_type"

    def queue_vector(self) -> dict[str, float]:
        queue: dict[str, float] = defaultdict(float)
        for runtime in self.vessels.values():
            if runtime.stage in {"not_arrived", "completed"}:
                continue
            stage = _queue_stage(runtime.stage)
            remaining = self._remaining_for_stage(runtime, stage)
            age = max(0.0, self.clock - runtime.spec.arrival_time)
            age_term = min(age, 240.0) / 240.0
            queue[_workload_key(stage, runtime.spec.cargo_class)] += runtime.spec.priority_weight * (
                remaining + age_term
            )
        return dict(sorted(queue.items()))

    def current_lower_service(self) -> dict[str, float]:
        service: dict[str, float] = defaultdict(float)
        for runtime in self.vessels.values():
            rate = self._runtime_lower_rate(runtime)
            if rate > 0.0:
                service[_workload_key(runtime.stage, runtime.spec.cargo_class)] += rate
        return dict(sorted(service.items()))

    def _dispatch_at_current_time(self) -> None:
        for _ in range(12):
            actions = self.feasible_atomic_actions()
            queue = self.queue_vector()
            base_service = self.current_lower_service()
            selection = _select_configuration(actions, queue, base_service, self.policy)
            audit = self._configuration_audit(selection, actions, queue, base_service)
            self.decision_audits.append(audit)
            if not selection.actions:
                return
            for action in selection.actions:
                ok, reason = self.check_action_feasible(action)
                self.feasibility_checks += 1
                if not ok:
                    self.feasibility_violations.append(f"{action.action_id}:{reason}")
                    raise PortSimulationError(f"selected action became infeasible: {reason}")
                self._apply_action(action)
            self._record_event(
                "dispatch",
                policy=self.policy,
                selected_action_ids=[action.action_id for action in selection.actions],
                robust_score=selection.robust_score,
                oracle_gap_alpha0=selection.oracle_gap_alpha0,
            )
        raise PortSimulationError(self._deadlock_message("same-time dispatch round limit exceeded"))

    def _configuration_audit(
        self,
        selection: ConfigurationSelection,
        actions: Sequence[AtomicAction],
        queue: Mapping[str, float],
        base_service: Mapping[str, float],
    ) -> dict[str, Any]:
        selected_ids = [row.action_id for row in selection.actions]
        payload = {
            "decision_time": _rounded(self.clock),
            "policy": self.policy,
            "queue_vector": _round_mapping(queue),
            "base_lower_service": _round_mapping(base_service),
            "candidate_action_count": len(actions),
            "candidate_configuration_count": selection.candidate_configuration_count,
            "selected_action_ids": selected_ids,
            "selected_actions": [row.snapshot() for row in selection.actions],
            "lower_service": _round_mapping(selection.lower_service),
            "penalty_units": _rounded(selection.penalty_units),
            "score": _rounded(selection.robust_score),
            "policy_objective": _rounded(selection.objective),
            "oracle_robust_score": _rounded(selection.oracle_robust_score),
            "oracle_gap_alpha0": _rounded(selection.oracle_gap_alpha0),
            "oracle_gap_alpha1": 0.0,
            "score_semantics": "robust_maxweight_lower_service",
            "bounded_penalty": True,
            "statewise_feasible_family": True,
            "finite_candidate_family": True,
            "exact_resource_mask_dp": True,
            "theorem_ready": self.policy == THEOREM_POLICY and selection.oracle_gap_alpha0 <= 1e-8,
        }
        payload["state_hash"] = _digest(
            {
                "time": payload["decision_time"],
                "queue": payload["queue_vector"],
                "service": payload["base_lower_service"],
                "resources": self._resource_snapshot(),
            }
        )
        return payload

    def _quay_start_actions(self) -> list[AtomicAction]:
        out: list[AtomicAction] = []
        idle_cranes = [key for key, value in self.crane_assignment.items() if value is None]
        for runtime in self.vessels.values():
            if runtime.stage != "waiting_quay":
                continue
            for berth in self.instance.berths:
                if self.berth_occupancy[berth.berth_id] is not None:
                    continue
                if not self._vessel_fits_berth(runtime.spec, berth):
                    continue
                compatible = [
                    crane_id
                    for crane_id in idle_cranes
                    if berth.berth_id in self.cranes[crane_id].compatible_berths
                ]
                for bundle in self._crane_bundles(runtime.spec, berth.berth_id, compatible):
                    nominal = self._quay_rate(runtime, berth.berth_id, bundle, lower=False)
                    lower = nominal * self.instance.lower_service_factor
                    penalty = 0.15 if berth.berth_id != runtime.spec.preferred_berth else 0.0
                    out.append(
                        AtomicAction(
                            action_id=f"start_quay|{runtime.spec.vessel_id}|{berth.berth_id}|{'+'.join(bundle)}",
                            action_type="start_quay",
                            vessel_id=runtime.spec.vessel_id,
                            stage="quay",
                            conflict_tokens=tuple(sorted(
                                (f"vessel:{runtime.spec.vessel_id}", f"berth:{berth.berth_id}")
                                + tuple(f"crane:{key}" for key in bundle)
                            )),
                            lower_service_delta=((_workload_key("quay", runtime.spec.cargo_class), lower),),
                            penalty_units=penalty,
                            nominal_rate=nominal,
                            lower_rate=lower,
                            estimated_duration=runtime.remaining_quay / lower,
                            metadata={
                                "berth_id": berth.berth_id,
                                "crane_ids": bundle,
                                "arrival_time": runtime.spec.arrival_time,
                                "due_time": runtime.spec.due_time,
                            },
                        )
                    )
        return out

    def _yard_start_actions(self) -> list[AtomicAction]:
        out: list[AtomicAction] = []
        for runtime in self.vessels.values():
            if runtime.stage != "waiting_yard":
                continue
            for block in self.instance.yard_blocks:
                nominal = block.rate(runtime.spec.cargo_class)
                if self.yard_occupancy[block.block_id] is not None or nominal <= 0.0:
                    continue
                lower = nominal * self.instance.lower_service_factor
                penalty = 0.10 if block.block_id != runtime.spec.preferred_yard else 0.0
                out.append(
                    AtomicAction(
                        action_id=f"start_yard|{runtime.spec.vessel_id}|{block.block_id}",
                        action_type="start_yard",
                        vessel_id=runtime.spec.vessel_id,
                        stage="yard",
                        conflict_tokens=(f"vessel:{runtime.spec.vessel_id}", f"yard:{block.block_id}"),
                        lower_service_delta=((_workload_key("yard", runtime.spec.cargo_class), lower),),
                        penalty_units=penalty,
                        nominal_rate=nominal,
                        lower_rate=lower,
                        estimated_duration=runtime.remaining_yard / lower,
                        metadata={
                            "yard_block_id": block.block_id,
                            "arrival_time": runtime.spec.arrival_time,
                            "due_time": runtime.spec.due_time,
                        },
                    )
                )
        return out

    def _gate_start_actions(self) -> list[AtomicAction]:
        out: list[AtomicAction] = []
        for runtime in self.vessels.values():
            if runtime.stage != "waiting_gate":
                continue
            for lane in self.instance.gate_lanes:
                nominal = lane.rate(runtime.spec.cargo_class)
                if self.gate_occupancy[lane.lane_id] is not None or nominal <= 0.0:
                    continue
                lower = nominal * self.instance.lower_service_factor
                out.append(
                    AtomicAction(
                        action_id=f"start_gate|{runtime.spec.vessel_id}|{lane.lane_id}",
                        action_type="start_gate",
                        vessel_id=runtime.spec.vessel_id,
                        stage="gate",
                        conflict_tokens=(f"vessel:{runtime.spec.vessel_id}", f"gate:{lane.lane_id}"),
                        lower_service_delta=((_workload_key("gate", runtime.spec.cargo_class), lower),),
                        penalty_units=0.0,
                        nominal_rate=nominal,
                        lower_rate=lower,
                        estimated_duration=runtime.remaining_gate / lower,
                        metadata={
                            "gate_lane_id": lane.lane_id,
                            "arrival_time": runtime.spec.arrival_time,
                            "due_time": runtime.spec.due_time,
                        },
                    )
                )
        return out

    def _reberth_actions(self) -> list[AtomicAction]:
        out: list[AtomicAction] = []
        idle_cranes = {key for key, value in self.crane_assignment.items() if value is None}
        for runtime in self.vessels.values():
            if runtime.stage != "quay" or runtime.berth_id is None:
                continue
            if runtime.reberth_count >= self.instance.max_reberths_per_vessel:
                continue
            progress = 1.0 - runtime.remaining_quay / runtime.spec.quay_work
            if progress < 0.10 or progress > 0.85:
                continue
            old_lower = self._runtime_lower_rate(runtime)
            old_completion = runtime.remaining_quay / max(old_lower, EPS)
            source = self.berths[runtime.berth_id]
            for target in self.instance.berths:
                if target.berth_id == source.berth_id or self.berth_occupancy[target.berth_id] is not None:
                    continue
                if not self._vessel_fits_berth(runtime.spec, target):
                    continue
                available = sorted(set(runtime.crane_ids) | idle_cranes)
                compatible = [key for key in available if target.berth_id in self.cranes[key].compatible_berths]
                bundles = self._crane_bundles(runtime.spec, target.berth_id, compatible)
                for bundle in bundles[-2:]:
                    nominal = self._quay_rate(runtime, target.berth_id, bundle, lower=False)
                    lower = nominal * self.instance.lower_service_factor
                    duration = self._reberth_duration(source, target)
                    fixed_cost = self.instance.reberth_fixed_cost
                    gain = old_completion - (duration + runtime.remaining_quay / max(lower, EPS))
                    if gain <= fixed_cost * self.instance.cost_to_penalty + EPS:
                        continue
                    key = _workload_key("quay", runtime.spec.cargo_class)
                    out.append(
                        AtomicAction(
                            action_id=f"reberth|{runtime.spec.vessel_id}|{source.berth_id}->{target.berth_id}|{'+'.join(bundle)}",
                            action_type="reberth",
                            vessel_id=runtime.spec.vessel_id,
                            stage="quay",
                            conflict_tokens=tuple(sorted(
                                (
                                    f"vessel:{runtime.spec.vessel_id}",
                                    f"berth:{source.berth_id}",
                                    f"berth:{target.berth_id}",
                                    "port:tug",
                                )
                                + tuple(f"crane:{crane}" for crane in bundle)
                            )),
                            lower_service_delta=((key, lower - old_lower),),
                            penalty_units=duration + fixed_cost * self.instance.cost_to_penalty,
                            nominal_rate=nominal,
                            lower_rate=lower,
                            estimated_duration=duration + runtime.remaining_quay / lower,
                            estimated_gain_s=gain,
                            fixed_cost=fixed_cost,
                            metadata={
                                "source_berth": source.berth_id,
                                "target_berth": target.berth_id,
                                "target_cranes": bundle,
                                "transition_duration": duration,
                                "progress_fraction": progress,
                            },
                        )
                    )
        return out

    def _crane_reassignment_actions(self) -> list[AtomicAction]:
        out: list[AtomicAction] = []
        for target in self.vessels.values():
            if target.stage != "quay" or target.berth_id is None:
                continue
            if len(target.crane_ids) >= target.spec.max_cranes:
                continue
            if target.crane_reassignment_count >= self.instance.max_crane_reassignments_per_vessel:
                continue
            progress = 1.0 - target.remaining_quay / target.spec.quay_work
            if progress < 0.08 or progress > 0.82:
                continue
            old_target_rate = self._runtime_lower_rate(target)
            candidates: list[tuple[str, str | None]] = []
            for crane_id, owner in self.crane_assignment.items():
                if crane_id in target.crane_ids or target.berth_id not in self.cranes[crane_id].compatible_berths:
                    continue
                if owner is None:
                    candidates.append((crane_id, None))
                elif owner in self.vessels:
                    source = self.vessels[owner]
                    if source.stage == "quay" and len(source.crane_ids) >= 2:
                        candidates.append((crane_id, owner))
            for crane_id, source_id in candidates:
                source = self.vessels[source_id] if source_id else None
                source_berth = self.berths[source.berth_id] if source and source.berth_id else self.berths[target.berth_id]
                target_berth = self.berths[target.berth_id]
                duration = self.instance.crane_reassignment_base_duration + (
                    abs(source_berth.position - target_berth.position)
                    * self.instance.crane_reassignment_distance_duration
                )
                new_target_cranes = tuple(sorted(target.crane_ids + [crane_id]))
                new_target_rate = self._quay_rate(target, target.berth_id, new_target_cranes, lower=True)
                old_total = target.remaining_quay / max(old_target_rate, EPS)
                delta: dict[str, float] = {
                    _workload_key("quay", target.spec.cargo_class): new_target_rate - old_target_rate
                }
                if source is not None:
                    old_source_rate = self._runtime_lower_rate(source)
                    new_source_cranes = tuple(key for key in source.crane_ids if key != crane_id)
                    new_source_rate = self._quay_rate(source, source.berth_id or "", new_source_cranes, lower=True)
                    old_total += source.remaining_quay / max(old_source_rate, EPS)
                    new_total = (
                        duration
                        + target.remaining_quay / max(new_target_rate, EPS)
                        + source.remaining_quay / max(new_source_rate, EPS)
                    )
                    source_key = _workload_key("quay", source.spec.cargo_class)
                    delta[source_key] = delta.get(source_key, 0.0) + new_source_rate - old_source_rate
                else:
                    new_total = duration + target.remaining_quay / max(new_target_rate, EPS)
                gain = old_total - new_total
                fixed_cost = self.instance.crane_reassignment_fixed_cost
                if gain <= fixed_cost * self.instance.cost_to_penalty + EPS:
                    continue
                tokens = [
                    f"vessel:{target.spec.vessel_id}",
                    f"crane:{crane_id}",
                    f"berth:{target.berth_id}",
                ]
                if source is not None:
                    tokens.extend((f"vessel:{source.spec.vessel_id}", f"berth:{source.berth_id}"))
                out.append(
                    AtomicAction(
                        action_id=f"crane_reassignment|{crane_id}|{source_id or 'idle'}->{target.spec.vessel_id}",
                        action_type="crane_reassignment",
                        vessel_id=target.spec.vessel_id,
                        stage="quay",
                        conflict_tokens=tuple(sorted(tokens)),
                        lower_service_delta=tuple(sorted(delta.items())),
                        penalty_units=duration + fixed_cost * self.instance.cost_to_penalty,
                        nominal_rate=new_target_rate / self.instance.lower_service_factor,
                        lower_rate=new_target_rate,
                        estimated_duration=duration + target.remaining_quay / new_target_rate,
                        estimated_gain_s=gain,
                        fixed_cost=fixed_cost,
                        metadata={
                            "crane_id": crane_id,
                            "source_vessel": source_id,
                            "target_vessel": target.spec.vessel_id,
                            "transition_duration": duration,
                            "progress_fraction": progress,
                        },
                    )
                )
        return out

    def _yard_rehandle_actions(self) -> list[AtomicAction]:
        if self.transfer_occupancy is not None:
            return []
        out: list[AtomicAction] = []
        for runtime in self.vessels.values():
            if runtime.stage != "yard" or runtime.yard_block_id is None:
                continue
            if runtime.yard_rehandle_count >= self.instance.max_yard_rehandles_per_vessel:
                continue
            progress = 1.0 - runtime.remaining_yard / runtime.spec.yard_work
            if progress < 0.10 or progress > 0.80:
                continue
            source = self.yards[runtime.yard_block_id]
            old_lower = source.rate(runtime.spec.cargo_class) * self.instance.lower_service_factor
            old_completion = runtime.remaining_yard / max(old_lower, EPS)
            for target in self.instance.yard_blocks:
                if target.block_id == source.block_id or self.yard_occupancy[target.block_id] is not None:
                    continue
                nominal = target.rate(runtime.spec.cargo_class)
                if nominal <= 0.0:
                    continue
                lower = nominal * self.instance.lower_service_factor
                duration = self.instance.yard_rehandle_base_duration + (
                    abs(source.position - target.position) * self.instance.yard_rehandle_distance_duration
                )
                fixed_cost = self.instance.yard_rehandle_fixed_cost
                gain = old_completion - (duration + runtime.remaining_yard / lower)
                if gain <= fixed_cost * self.instance.cost_to_penalty + EPS:
                    continue
                key = _workload_key("yard", runtime.spec.cargo_class)
                out.append(
                    AtomicAction(
                        action_id=f"yard_rehandle|{runtime.spec.vessel_id}|{source.block_id}->{target.block_id}",
                        action_type="yard_rehandle",
                        vessel_id=runtime.spec.vessel_id,
                        stage="yard",
                        conflict_tokens=tuple(sorted((
                            f"vessel:{runtime.spec.vessel_id}",
                            f"yard:{source.block_id}",
                            f"yard:{target.block_id}",
                            "yard_transfer",
                        ))),
                        lower_service_delta=((key, lower - old_lower),),
                        penalty_units=duration + fixed_cost * self.instance.cost_to_penalty,
                        nominal_rate=nominal,
                        lower_rate=lower,
                        estimated_duration=duration + runtime.remaining_yard / lower,
                        estimated_gain_s=gain,
                        fixed_cost=fixed_cost,
                        metadata={
                            "source_yard": source.block_id,
                            "target_yard": target.block_id,
                            "transition_duration": duration,
                            "progress_fraction": progress,
                        },
                    )
                )
        return out

    def _apply_action(self, action: AtomicAction) -> None:
        runtime = self.vessels[action.vessel_id]
        meta = action.metadata
        self.action_counts[action.action_type] += 1
        if action.action_type == "start_quay":
            berth = str(meta["berth_id"])
            cranes = [str(item) for item in meta["crane_ids"]]
            runtime.stage = "quay"
            runtime.berth_id = berth
            runtime.crane_ids = cranes
            runtime.quay_start = self.clock if runtime.quay_start is None else runtime.quay_start
            self.berth_occupancy[berth] = runtime.spec.vessel_id
            for crane in cranes:
                self.crane_assignment[crane] = runtime.spec.vessel_id
        elif action.action_type == "start_yard":
            block = str(meta["yard_block_id"])
            runtime.stage = "yard"
            runtime.yard_block_id = block
            runtime.yard_start = self.clock if runtime.yard_start is None else runtime.yard_start
            self.yard_occupancy[block] = runtime.spec.vessel_id
        elif action.action_type == "start_gate":
            lane = str(meta["gate_lane_id"])
            runtime.stage = "gate"
            runtime.gate_lane_id = lane
            runtime.gate_start = self.clock if runtime.gate_start is None else runtime.gate_start
            self.gate_occupancy[lane] = runtime.spec.vessel_id
        elif action.action_type == "reberth":
            self._apply_reberth(runtime, action)
        elif action.action_type == "crane_reassignment":
            self._apply_crane_reassignment(runtime, action)
        elif action.action_type == "yard_rehandle":
            self._apply_yard_rehandle(runtime, action)
        else:
            raise PortSimulationError(f"unsupported action type: {action.action_type}")
        if action.action_type in RECONFIGURATION_ACTIONS:
            self.reconfiguration_duration += float(meta["transition_duration"])
            self.reconfiguration_fixed_cost += action.fixed_cost

    def _apply_reberth(self, runtime: VesselRuntime, action: AtomicAction) -> None:
        meta = action.metadata
        source = str(meta["source_berth"])
        target = str(meta["target_berth"])
        cranes = [str(item) for item in meta["target_cranes"]]
        marker = f"transition:{runtime.spec.vessel_id}:reberth"
        self.berth_occupancy[source] = marker
        self.berth_occupancy[target] = marker
        for crane in list(runtime.crane_ids):
            self.crane_assignment[crane] = None
        for crane in cranes:
            self.crane_assignment[crane] = marker
        runtime.stage = "reberthing"
        runtime.berth_id = None
        runtime.crane_ids = []
        runtime.reberth_count += 1
        self.transitions.append(
            PendingTransition(
                "reberth",
                runtime.spec.vessel_id,
                self.clock + float(meta["transition_duration"]),
                {"source_berth": source, "target_berth": target, "target_cranes": cranes},
            )
        )

    def _apply_crane_reassignment(self, runtime: VesselRuntime, action: AtomicAction) -> None:
        meta = action.metadata
        crane = str(meta["crane_id"])
        source_id = str(meta.get("source_vessel") or "")
        if source_id:
            source = self.vessels[source_id]
            source.crane_ids.remove(crane)
        marker = f"transition:{runtime.spec.vessel_id}:crane"
        self.crane_assignment[crane] = marker
        runtime.crane_reassignment_count += 1
        self.transitions.append(
            PendingTransition(
                "crane_reassignment",
                runtime.spec.vessel_id,
                self.clock + float(meta["transition_duration"]),
                {"crane_id": crane, "source_vessel": source_id},
            )
        )

    def _apply_yard_rehandle(self, runtime: VesselRuntime, action: AtomicAction) -> None:
        meta = action.metadata
        source = str(meta["source_yard"])
        target = str(meta["target_yard"])
        marker = f"transition:{runtime.spec.vessel_id}:yard"
        self.yard_occupancy[source] = marker
        self.yard_occupancy[target] = marker
        self.transfer_occupancy = marker
        runtime.stage = "rehandling"
        runtime.yard_block_id = None
        runtime.yard_rehandle_count += 1
        self.transitions.append(
            PendingTransition(
                "yard_rehandle",
                runtime.spec.vessel_id,
                self.clock + float(meta["transition_duration"]),
                {"source_yard": source, "target_yard": target},
            )
        )

    def _next_event_time(self) -> float:
        candidates: list[float] = []
        future_arrivals = [
            runtime.spec.arrival_time
            for runtime in self.vessels.values()
            if runtime.stage == "not_arrived" and runtime.spec.arrival_time > self.clock + EPS
        ]
        candidates.extend(future_arrivals)
        candidates.extend(row.complete_at for row in self.transitions if row.complete_at > self.clock + EPS)
        for runtime in self.vessels.values():
            rate = self._runtime_nominal_rate(runtime)
            if rate > 0.0:
                remaining = self._remaining_for_stage(runtime, runtime.stage)
                candidates.append(self.clock + remaining / rate)
        if not self._all_completed():
            while self.next_decision_time <= self.clock + EPS:
                self.next_decision_time += self.instance.decision_interval
            candidates.append(self.next_decision_time)
        return min(candidates) if candidates else math.inf

    def _advance_to(self, next_time: float) -> None:
        dt = next_time - self.clock
        if dt <= 0.0:
            raise PortSimulationError("advance duration must be positive")
        for runtime in self.vessels.values():
            rate = self._runtime_nominal_rate(runtime)
            if rate <= 0.0:
                continue
            if runtime.stage == "quay":
                runtime.remaining_quay = max(0.0, runtime.remaining_quay - rate * dt)
            elif runtime.stage == "yard":
                runtime.remaining_yard = max(0.0, runtime.remaining_yard - rate * dt)
            elif runtime.stage == "gate":
                runtime.remaining_gate = max(0.0, runtime.remaining_gate - rate * dt)
        for key, value in self.berth_occupancy.items():
            if value is not None:
                self.resource_busy_time[f"berth:{key}"] += dt
        for key, value in self.crane_assignment.items():
            if value is not None:
                self.resource_busy_time[f"crane:{key}"] += dt
        for key, value in self.yard_occupancy.items():
            if value is not None:
                self.resource_busy_time[f"yard:{key}"] += dt
        for key, value in self.gate_occupancy.items():
            if value is not None:
                self.resource_busy_time[f"gate:{key}"] += dt
        if self.transfer_occupancy is not None:
            self.resource_busy_time["yard_transfer"] += dt
        self.clock = next_time
        active = sum(runtime.stage not in {"not_arrived", "completed"} for runtime in self.vessels.values())
        self._peak_active_vessels = max(self._peak_active_vessels, active)

    def _process_events_at_clock(self) -> None:
        completed_transitions = [row for row in self.transitions if row.complete_at <= self.clock + EPS]
        self.transitions = [row for row in self.transitions if row.complete_at > self.clock + EPS]
        for transition in sorted(completed_transitions, key=lambda row: (row.complete_at, row.kind, row.vessel_id)):
            self._complete_transition(transition)
        for runtime in sorted(self.vessels.values(), key=lambda row: row.spec.vessel_id):
            if runtime.stage == "quay" and runtime.remaining_quay <= EPS:
                self._complete_quay(runtime)
            elif runtime.stage == "yard" and runtime.remaining_yard <= EPS:
                self._complete_yard(runtime)
            elif runtime.stage == "gate" and runtime.remaining_gate <= EPS:
                self._complete_gate(runtime)
        self._release_arrivals()
        if abs(self.clock - self.next_decision_time) <= EPS:
            self._record_event("control_tick")

    def _complete_transition(self, transition: PendingTransition) -> None:
        runtime = self.vessels[transition.vessel_id]
        meta = transition.metadata
        if transition.kind == "reberth":
            source = str(meta["source_berth"])
            target = str(meta["target_berth"])
            cranes = [str(item) for item in meta["target_cranes"]]
            self.berth_occupancy[source] = None
            self.berth_occupancy[target] = runtime.spec.vessel_id
            for crane in cranes:
                self.crane_assignment[crane] = runtime.spec.vessel_id
            runtime.stage = "quay"
            runtime.berth_id = target
            runtime.crane_ids = cranes
        elif transition.kind == "crane_reassignment":
            crane = str(meta["crane_id"])
            if runtime.stage != "quay":
                self.crane_assignment[crane] = None
            else:
                self.crane_assignment[crane] = runtime.spec.vessel_id
                runtime.crane_ids.append(crane)
                runtime.crane_ids.sort()
        elif transition.kind == "yard_rehandle":
            source = str(meta["source_yard"])
            target = str(meta["target_yard"])
            self.yard_occupancy[source] = None
            self.yard_occupancy[target] = runtime.spec.vessel_id
            self.transfer_occupancy = None
            runtime.stage = "yard"
            runtime.yard_block_id = target
        else:
            raise PortSimulationError(f"unknown transition kind: {transition.kind}")
        self._record_event("transition_complete", kind=transition.kind, vessel_id=transition.vessel_id)

    def _complete_quay(self, runtime: VesselRuntime) -> None:
        if runtime.berth_id is not None:
            self.berth_occupancy[runtime.berth_id] = None
        for crane in runtime.crane_ids:
            self.crane_assignment[crane] = None
        runtime.remaining_quay = 0.0
        runtime.stage = "waiting_yard"
        runtime.berth_id = None
        runtime.crane_ids = []
        self._record_event("stage_complete", vessel_id=runtime.spec.vessel_id, stage="quay")

    def _complete_yard(self, runtime: VesselRuntime) -> None:
        if runtime.yard_block_id is not None:
            self.yard_occupancy[runtime.yard_block_id] = None
        runtime.remaining_yard = 0.0
        runtime.stage = "waiting_gate"
        runtime.yard_block_id = None
        self._record_event("stage_complete", vessel_id=runtime.spec.vessel_id, stage="yard")

    def _complete_gate(self, runtime: VesselRuntime) -> None:
        if runtime.gate_lane_id is not None:
            self.gate_occupancy[runtime.gate_lane_id] = None
        runtime.remaining_gate = 0.0
        runtime.stage = "completed"
        runtime.gate_lane_id = None
        runtime.completion_time = self.clock
        self._record_event("vessel_complete", vessel_id=runtime.spec.vessel_id)

    def _release_arrivals(self) -> None:
        for runtime in sorted(self.vessels.values(), key=lambda row: (row.spec.arrival_time, row.spec.vessel_id)):
            if runtime.stage == "not_arrived" and runtime.spec.arrival_time <= self.clock + EPS:
                runtime.stage = "waiting_quay"
                self._record_event("arrival", vessel_id=runtime.spec.vessel_id)

    def _runtime_nominal_rate(self, runtime: VesselRuntime) -> float:
        if runtime.stage == "quay" and runtime.berth_id and runtime.crane_ids:
            return self._quay_rate(runtime, runtime.berth_id, runtime.crane_ids, lower=False)
        if runtime.stage == "yard" and runtime.yard_block_id:
            return self.yards[runtime.yard_block_id].rate(runtime.spec.cargo_class)
        if runtime.stage == "gate" and runtime.gate_lane_id:
            return self.gates[runtime.gate_lane_id].rate(runtime.spec.cargo_class)
        return 0.0

    def _runtime_lower_rate(self, runtime: VesselRuntime) -> float:
        return self._runtime_nominal_rate(runtime) * self.instance.lower_service_factor

    def _quay_rate(
        self,
        runtime: VesselRuntime,
        berth_id: str,
        crane_ids: Iterable[str],
        *,
        lower: bool,
    ) -> float:
        cranes = sorted(
            (self.cranes[key] for key in crane_ids),
            key=lambda row: (-row.base_rate * row.cargo_multiplier(runtime.spec.cargo_class), row.crane_id),
        )
        discounts = (1.0, 0.86, 0.72, 0.60)
        raw = sum(
            crane.base_rate * crane.cargo_multiplier(runtime.spec.cargo_class) * discounts[index]
            for index, crane in enumerate(cranes)
        )
        rate = raw * self.berths[berth_id].rate_multiplier
        return rate * self.instance.lower_service_factor if lower else rate

    def _crane_bundles(
        self,
        vessel: VesselSpec,
        berth_id: str,
        crane_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], ...]:
        compatible = sorted(
            (key for key in crane_ids if berth_id in self.cranes[key].compatible_berths),
            key=lambda key: (-self.cranes[key].base_rate * self.cranes[key].cargo_multiplier(vessel.cargo_class), key),
        )
        if not compatible:
            return ()
        bundles: set[tuple[str, ...]] = {(key,) for key in compatible}
        for size in range(2, min(vessel.max_cranes, len(compatible)) + 1):
            bundles.add(tuple(sorted(compatible[:size])))
        return tuple(sorted(bundles, key=lambda row: (len(row), row)))

    def _reberth_duration(self, source: BerthSpec, target: BerthSpec) -> float:
        return self.instance.reberth_base_duration + (
            abs(source.position - target.position) * self.instance.reberth_distance_duration
        )

    @staticmethod
    def _vessel_fits_berth(vessel: VesselSpec, berth: BerthSpec) -> bool:
        return vessel.length <= berth.max_length and vessel.draft <= berth.max_draft

    @staticmethod
    def _remaining_for_stage(runtime: VesselRuntime, stage: str) -> float:
        if stage == "quay":
            return runtime.remaining_quay
        if stage == "yard":
            return runtime.remaining_yard
        if stage == "gate":
            return runtime.remaining_gate
        return 0.0

    def _all_completed(self) -> bool:
        return all(runtime.stage == "completed" for runtime in self.vessels.values())

    def _record_event(self, event: str, **values: Any) -> None:
        self.event_trace.append({"time": _rounded(self.clock), "event": event, **_canonical(values)})

    def _resource_snapshot(self) -> dict[str, Any]:
        return {
            "berths": dict(sorted(self.berth_occupancy.items())),
            "cranes": dict(sorted(self.crane_assignment.items())),
            "yards": dict(sorted(self.yard_occupancy.items())),
            "gates": dict(sorted(self.gate_occupancy.items())),
            "yard_transfer": self.transfer_occupancy,
        }

    def _result(self) -> dict[str, Any]:
        completion_rows = []
        for runtime in sorted(self.vessels.values(), key=lambda row: row.spec.vessel_id):
            if runtime.completion_time is None or runtime.quay_start is None:
                raise PortSimulationError(f"incomplete timestamp ledger for {runtime.spec.vessel_id}")
            completion_rows.append(
                {
                    "vessel_id": runtime.spec.vessel_id,
                    "arrival_time": _rounded(runtime.spec.arrival_time),
                    "due_time": _rounded(runtime.spec.due_time),
                    "completion_time": _rounded(runtime.completion_time),
                    "flow_time": _rounded(runtime.completion_time - runtime.spec.arrival_time),
                    "berth_wait": _rounded(runtime.quay_start - runtime.spec.arrival_time),
                    "tardiness": _rounded(max(0.0, runtime.completion_time - runtime.spec.due_time)),
                    "priority_weight": _rounded(runtime.spec.priority_weight),
                    "reberth_count": runtime.reberth_count,
                    "crane_reassignment_count": runtime.crane_reassignment_count,
                    "yard_rehandle_count": runtime.yard_rehandle_count,
                }
            )
        flows = [float(row["flow_time"]) for row in completion_rows]
        waits = [float(row["berth_wait"]) for row in completion_rows]
        arrivals = [runtime.spec.arrival_time for runtime in self.vessels.values()]
        horizon = self.clock - min(arrivals)
        total_work = sum(
            row.spec.quay_work + row.spec.yard_work + row.spec.gate_work
            for row in self.vessels.values()
        )
        weighted_tardiness = sum(
            float(row["tardiness"]) * float(row["priority_weight"])
            for row in completion_rows
        )
        utilization = {
            key: _rounded(value / max(horizon, EPS))
            for key, value in sorted(self.resource_busy_time.items())
        }
        theorem_slots = [row for row in self.decision_audits if row["policy"] == THEOREM_POLICY]
        metrics = {
            "makespan": _rounded(horizon),
            "mean_flow_time": _rounded(sum(flows) / len(flows)),
            "p95_flow_time": _rounded(_quantile(flows, 0.95)),
            "mean_berth_wait": _rounded(sum(waits) / len(waits)),
            "weighted_tardiness": _rounded(weighted_tardiness),
            "throughput_work_per_time": _rounded(total_work / max(horizon, EPS)),
            "completed_vessels": len(completion_rows),
            "reconfiguration_count": sum(self.action_counts[key] for key in RECONFIGURATION_ACTIONS),
            "reconfiguration_duration": _rounded(self.reconfiguration_duration),
            "reconfiguration_fixed_cost": _rounded(self.reconfiguration_fixed_cost),
            "feasibility_violation_count": len(self.feasibility_violations),
            "event_count": len(self.event_trace),
            "decision_slot_count": len(self.decision_audits),
            "peak_active_vessels": self._peak_active_vessels,
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "instance": self.instance.name,
            "instance_digest": self.instance.digest,
            "policy": self.policy,
            "status": "PORT_SIMULATION_PASS" if not self.feasibility_violations else "PORT_SIMULATION_INVALID",
            "pass": not self.feasibility_violations and len(completion_rows) == len(self.vessels),
            "metrics": metrics,
            "resource_utilization": utilization,
            "action_counts": dict(sorted(self.action_counts.items())),
            "vessels": completion_rows,
            "decision_audits": self.decision_audits,
            "event_trace": self.event_trace,
            "theorem_contract": {
                "score_semantics": "robust_maxweight_lower_service",
                "finite_statewise_candidate_family": True,
                "lower_service_source": "registered_instance_conservative_rate",
                "lower_service_factor": self.instance.lower_service_factor,
                "bounded_reconfiguration_penalty": True,
                "exact_finite_family_oracle": self.policy == THEOREM_POLICY and all(
                    float(row["oracle_gap_alpha0"]) <= 1e-8 for row in theorem_slots
                ),
                "oracle_gap_alpha1": 0.0,
            },
            "feasibility_audit": {
                "checks": self.feasibility_checks,
                "violations": list(self.feasibility_violations),
                "all_selected_actions_feasible": not self.feasibility_violations,
            },
        }
        result["result_hash"] = _digest(
            {
                "instance": result["instance"],
                "instance_digest": result["instance_digest"],
                "policy": result["policy"],
                "metrics": result["metrics"],
                "resources": result["resource_utilization"],
                "actions": result["action_counts"],
                "vessels": result["vessels"],
                "decisions": result["decision_audits"],
                "events": result["event_trace"],
            }
        )
        return result

    def _deadlock_message(self, reason: str) -> str:
        states = Counter(runtime.stage for runtime in self.vessels.values())
        return f"{reason}; time={self.clock:.6f}; stages={dict(states)}; resources={self._resource_snapshot()}"


def _select_configuration(
    actions: Sequence[AtomicAction],
    queue: Mapping[str, float],
    base_service: Mapping[str, float],
    policy: str,
) -> ConfigurationSelection:
    selected, objective, count = _resource_mask_dp(actions, queue, policy)
    oracle_actions, _, _ = _resource_mask_dp(actions, queue, THEOREM_POLICY)
    lower = _configuration_service(base_service, selected)
    oracle_lower = _configuration_service(base_service, oracle_actions)
    penalty = sum(action.penalty_units for action in selected)
    oracle_penalty = sum(action.penalty_units for action in oracle_actions)
    robust_score = _robust_score(lower, queue, penalty)
    oracle_score = _robust_score(oracle_lower, queue, oracle_penalty)
    return ConfigurationSelection(
        actions=selected,
        objective=objective,
        robust_score=robust_score,
        oracle_robust_score=oracle_score,
        candidate_configuration_count=count,
        lower_service=lower,
        penalty_units=penalty,
    )


def _resource_mask_dp(
    actions: Sequence[AtomicAction],
    queue: Mapping[str, float],
    policy: str,
) -> tuple[tuple[AtomicAction, ...], float, int]:
    eligible = [
        action for action in actions
        if action.theorem_ready and not (
            policy in {"fcfs_static", "spt_static", "edd_static"}
            and action.action_type in RECONFIGURATION_ACTIONS
        )
    ]
    tokens = sorted({token for action in eligible for token in action.conflict_tokens})
    bit = {token: 1 << index for index, token in enumerate(tokens)}
    masks = {
        action.action_id: sum(bit[token] for token in set(action.conflict_tokens))
        for action in eligible
    }
    groups: dict[str, list[AtomicAction]] = defaultdict(list)
    for action in eligible:
        groups[action.vessel_id].append(action)
    states: dict[int, tuple[float, tuple[AtomicAction, ...]]] = {0: (0.0, ())}
    counts: dict[int, int] = {0: 1}
    for vessel_id in sorted(groups):
        options = sorted(groups[vessel_id], key=lambda row: row.action_id)
        new_states: dict[int, tuple[float, tuple[AtomicAction, ...]]] = dict(states)
        new_counts: dict[int, int] = dict(counts)
        for used_mask, (score, chosen) in states.items():
            for action in options:
                action_mask = masks[action.action_id]
                if used_mask & action_mask:
                    continue
                new_mask = used_mask | action_mask
                candidate = tuple(sorted(chosen + (action,), key=lambda row: row.action_id))
                candidate_score = score + _policy_action_weight(action, queue, policy)
                current = new_states.get(new_mask)
                new_counts[new_mask] = new_counts.get(new_mask, 0) + counts[used_mask]
                if current is None or candidate_score > current[0] + EPS or (
                    abs(candidate_score - current[0]) <= EPS
                    and _action_ids(candidate) < _action_ids(current[1])
                ):
                    new_states[new_mask] = (candidate_score, candidate)
        states = new_states
        counts = new_counts
    best = min(
        states.values(),
        key=lambda row: (-row[0], _action_ids(row[1])),
    )
    configuration_count = sum(counts.values())
    return best[1], float(best[0]), int(configuration_count)


def _policy_action_weight(action: AtomicAction, queue: Mapping[str, float], policy: str) -> float:
    if policy == THEOREM_POLICY:
        return sum(float(queue.get(key, 0.0)) * value for key, value in action.lower_service_delta) - action.penalty_units
    arrival = float(action.metadata.get("arrival_time", 0.0))
    due = float(action.metadata.get("due_time", math.inf))
    if action.action_type in RECONFIGURATION_ACTIONS:
        return max(0.0, action.estimated_gain_s) if policy == "reconfiguration_greedy" else -math.inf
    if policy == "fcfs_static":
        return 1_000.0 + 1.0 / (1.0 + max(0.0, arrival))
    if policy == "spt_static":
        return 1_000.0 + 100.0 / max(action.estimated_duration, EPS)
    if policy == "edd_static":
        return 1_000.0 + 100.0 / max(1.0, due)
    if policy == "reconfiguration_greedy":
        return 1_000.0 + action.lower_rate + 10.0 / max(action.estimated_duration, EPS)
    raise ValueError(f"unknown policy: {policy}")


def _configuration_service(
    base_service: Mapping[str, float],
    actions: Sequence[AtomicAction],
) -> dict[str, float]:
    out = defaultdict(float, {str(key): float(value) for key, value in base_service.items()})
    for action in actions:
        for key, value in action.lower_service_delta:
            out[str(key)] += float(value)
    for key, value in list(out.items()):
        if value < -1e-7:
            raise PortSimulationError(f"configuration has negative lower service for {key}: {value}")
        out[key] = max(0.0, value)
    return dict(sorted(out.items()))


def _robust_score(service: Mapping[str, float], queue: Mapping[str, float], penalty: float) -> float:
    return sum(float(queue.get(key, 0.0)) * float(value) for key, value in service.items()) - float(penalty)


def run_port_instance(instance: PortInstance, policy: str) -> dict[str, Any]:
    return PortSchedulingSimulator(instance, policy).run()


def build_port_scheduling_benchmark(
    instances: Sequence[PortInstance] | None = None,
    policies: Sequence[str] = ALL_POLICIES,
    *,
    determinism_repeats: int = 2,
) -> dict[str, Any]:
    selected_instances = tuple(instances or built_in_port_instances())
    selected_policies = tuple(policies)
    if determinism_repeats < 2:
        raise ValueError("determinism_repeats must be at least two")
    if len(selected_policies) < 2 or THEOREM_POLICY not in selected_policies:
        raise ValueError("benchmark requires the theorem policy and at least one baseline")
    if any(policy not in ALL_POLICIES for policy in selected_policies):
        raise ValueError("benchmark contains an unknown policy")

    instance_reports: list[dict[str, Any]] = []
    all_deterministic = True
    all_valid = True
    metric_schemas: set[tuple[str, ...]] = set()
    for instance in selected_instances:
        policy_results: dict[str, Any] = {}
        determinism: dict[str, Any] = {}
        for policy in selected_policies:
            repeats = [run_port_instance(instance, policy) for _ in range(determinism_repeats)]
            hashes = [str(row["result_hash"]) for row in repeats]
            deterministic = len(set(hashes)) == 1
            all_deterministic = all_deterministic and deterministic
            all_valid = all_valid and all(bool(row["pass"]) for row in repeats)
            result = repeats[0]
            metric_schemas.add(tuple(sorted(result["metrics"])))
            policy_results[policy] = result
            determinism[policy] = {
                "repeat_count": determinism_repeats,
                "hashes": hashes,
                "deterministic": deterministic,
            }
        pareto = _instance_pareto(policy_results)
        instance_reports.append(
            {
                "instance": instance.name,
                "instance_digest": instance.digest,
                "resource_counts": {
                    "berths": len(instance.berths),
                    "quay_cranes": len(instance.quay_cranes),
                    "yard_blocks": len(instance.yard_blocks),
                    "gate_lanes": len(instance.gate_lanes),
                    "vessels": len(instance.vessels),
                },
                "policies": policy_results,
                "determinism": determinism,
                "pareto": pareto,
            }
        )

    aggregate = _aggregate_metrics(instance_reports, selected_policies)
    aggregate_pareto = _metrics_pareto(aggregate)
    theorem_exact = all(
        bool(row["policies"][THEOREM_POLICY]["theorem_contract"]["exact_finite_family_oracle"])
        for row in instance_reports
    )
    migration_analogues = {
        action
        for row in instance_reports
        for action, count in row["policies"][THEOREM_POLICY]["action_counts"].items()
        if action in RECONFIGURATION_ACTIONS and int(count) > 0
    }
    benchmark_valid = (
        all_valid
        and all_deterministic
        and len(metric_schemas) == 1
        and theorem_exact
        and len(selected_instances) >= 2
        and len(selected_policies) >= 4
        and migration_analogues == RECONFIGURATION_ACTIONS
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": "theorem_facing_port_scheduling_benchmark",
        "status": "PORT_BENCHMARK_PASS" if benchmark_valid else "PORT_BENCHMARK_OPEN",
        "pass": benchmark_valid,
        "scoped_claim_ready": benchmark_valid,
        "physical_port_claim_ready": False,
        "global_port_optimality_claim_ready": False,
        "deterministic_reproduction_ready": all_deterministic,
        "same_metric_schema_ready": len(metric_schemas) == 1,
        "statewise_feasibility_ready": all_valid,
        "exact_finite_family_oracle_ready": theorem_exact,
        "frame_bridge_ready": False,
        "registered_instance_count": len(selected_instances),
        "registered_vessel_count": sum(len(row.vessels) for row in selected_instances),
        "policies": list(selected_policies),
        "baseline_count": sum(policy != THEOREM_POLICY for policy in selected_policies),
        "migration_analogues_exercised": sorted(migration_analogues),
        "all_three_migration_analogues_exercised": migration_analogues == RECONFIGURATION_ACTIONS,
        "instances": instance_reports,
        "aggregate_metrics": aggregate,
        "aggregate_pareto": aggregate_pareto,
        "ours_pareto_nondominated": THEOREM_POLICY in aggregate_pareto["nondominated_policies"],
        "ours_strictly_dominates_all_baselines": _strictly_dominates_all(
            aggregate, THEOREM_POLICY, [row for row in selected_policies if row != THEOREM_POLICY]
        ),
        "theorem_contract": {
            "configuration_action": "finite statewise berth/quay-crane/yard/gate allocation plus bounded reconfiguration actions",
            "score": "Q^T lower_service - reconfiguration_penalty",
            "candidate_solver": "exact dominance-preserving resource-mask dynamic program",
            "oracle_gap_alpha0": 0.0 if theorem_exact else None,
            "oracle_gap_alpha1": 0.0,
            "lower_service_provenance": "registered synthetic rate multiplied by instance lower_service_factor",
            "dynamic_feasibility": "recomputed at every arrival, completion, control tick, and transition completion",
            "timing_semantics": (
                "event-driven semi-Markov transitions; transition downtime is realized in the clock and "
                "also exposed as a bounded decision penalty"
            ),
        },
        "claim_boundary": {
            "supports": [
                "deterministic event-driven replay on the registered heterogeneous port instances",
                "explicit berth, quay-crane, yard-block, gate-lane, tug, and yard-transfer feasibility",
                "bounded-duration and bounded-cost re-berthing, crane reassignment, and yard rehandle actions",
                "same-simulator comparison with FCFS, SPT, EDD, and reconfiguration-aware greedy heuristics",
                "finite-family robust MaxWeight action-ledger and exact oracle-gap audit",
            ],
            "does_not_support": [
                "measured performance or operational safety at a physical port",
                "optimality over the unrestricted berth-allocation or quay-crane scheduling action space",
                "superiority to every exact, decomposition, or commercial terminal scheduler",
                "stochastic throughput stability without a separately calibrated arrival/service model",
                "direct instantiation of the slotted-time Lean drift theorem without a frame or uniformization certificate",
                "generalization to unregistered future port layouts, weather, tides, labor rules, or vessel classes",
            ],
            "performance_wording": (
                "Report numerical superiority only for metrics and registered instances where the artifact records it; "
                "benchmark validity does not imply superiority."
            ),
        },
    }
    report["artifact_hash"] = _digest({key: value for key, value in report.items() if key != "artifact_hash"})
    return report


def _instance_pareto(policy_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {policy: row["metrics"] for policy, row in policy_results.items()}
    return _metrics_pareto(metrics)


def _metrics_pareto(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    nondominated = []
    dominance: dict[str, list[str]] = {}
    for policy, row in metrics.items():
        dominated_by = [
            other
            for other, other_row in metrics.items()
            if other != policy and _dominates(other_row, row)
        ]
        dominance[policy] = sorted(dominated_by)
        if not dominated_by:
            nondominated.append(policy)
    return {
        "primary_cost_metrics": list(PRIMARY_COST_METRICS),
        "nondominated_policies": sorted(nondominated),
        "dominated_by": dominance,
    }


def _aggregate_metrics(
    instance_reports: Sequence[Mapping[str, Any]],
    policies: Sequence[str],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for policy in policies:
        rows = [instance["policies"][policy]["metrics"] for instance in instance_reports]
        out[policy] = {
            metric: _rounded(sum(float(row[metric]) for row in rows) / len(rows))
            for metric in PRIMARY_COST_METRICS
        }
        out[policy]["mean_throughput_work_per_time"] = _rounded(
            sum(float(row["throughput_work_per_time"]) for row in rows) / len(rows)
        )
        out[policy]["total_reconfiguration_count"] = float(
            sum(int(row["reconfiguration_count"]) for row in rows)
        )
    return out


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = all(float(left[key]) <= float(right[key]) + 1e-8 for key in PRIMARY_COST_METRICS)
    strictly_better = any(float(left[key]) < float(right[key]) - 1e-8 for key in PRIMARY_COST_METRICS)
    return no_worse and strictly_better


def _strictly_dominates_all(
    metrics: Mapping[str, Mapping[str, Any]],
    policy: str,
    baselines: Sequence[str],
) -> bool:
    return bool(baselines) and all(_dominates(metrics[policy], metrics[baseline]) for baseline in baselines)


def _queue_stage(stage: str) -> str:
    if stage in {"waiting_quay", "quay", "reberthing"}:
        return "quay"
    if stage in {"waiting_yard", "yard", "rehandling"}:
        return "yard"
    if stage in {"waiting_gate", "gate"}:
        return "gate"
    return stage


def _workload_key(stage: str, cargo_class: str) -> str:
    return f"port:{_queue_stage(stage)}:{cargo_class}"


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * min(1.0, max(0.0, quantile))
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _action_ids(actions: Iterable[AtomicAction]) -> tuple[str, ...]:
    return tuple(action.action_id for action in actions)


def _rounded(value: float) -> float:
    return round(float(value), 9)


def _round_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): _rounded(float(value)) for key, value in sorted(values.items())}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        return _rounded(value)
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-json", action="append", type=Path, default=[])
    parser.add_argument("--policy", action="append", choices=ALL_POLICIES, default=[])
    parser.add_argument("--determinism-repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    instances = tuple(load_port_instance(path) for path in args.instance_json) or built_in_port_instances()
    policies = tuple(args.policy) or ALL_POLICIES
    report = build_port_scheduling_benchmark(
        instances,
        policies,
        determinism_repeats=args.determinism_repeats,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()


__all__ = [
    "ALL_POLICIES",
    "AtomicAction",
    "BASELINE_POLICIES",
    "PortSchedulingSimulator",
    "PortSimulationError",
    "THEOREM_POLICY",
    "build_port_scheduling_benchmark",
    "run_port_instance",
]
