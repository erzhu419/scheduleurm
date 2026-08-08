"""Auditable adapter for the public BACASP-S port benchmark.

The raw fixture is preserved byte-for-byte from the authors' repository.  This
module parses the public continuous-quay / specific-crane core and maps it to the
existing :class:`PortInstance` interface.  Fields absent from BACASP-S (draft,
yard, and gate) are explicitly recorded as schema augmentations.

The resulting run is a projected event-driven comparison on public input data.
It is not a reproduction of the paper's MILP, published objective values, or
continuous-quay optimum.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.port_scheduling_benchmark import (
    ALL_POLICIES,
    AtomicAction,
    PendingTransition,
    PortSchedulingSimulator,
    THEOREM_POLICY,
)
from algorithm.experiments.port_scheduling_instances import (
    BerthSpec,
    GateLaneSpec,
    PortInstance,
    QuayCraneSpec,
    VesselSpec,
    YardBlockSpec,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "data" / "port_public_bacasp_s_large_mb_30_fixture.dat"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "data" / "port_public_source_manifest.json"
SCHEMA_VERSION = "scheduleurm.port_public_bacasp_s_adapter.v1"
EPS = 1e-9
SOURCE_COST_METRICS = ("quay_makespan", "mean_quay_flow_time", "source_objective_cost")


@dataclass(frozen=True)
class BacaspSVesselRecord:
    vessel_index: int
    length: float
    arrival_time: float
    desired_departure: float
    position_cost: float
    waiting_cost: float
    delay_cost: float
    desired_position: float
    min_cranes: int
    max_cranes: int
    processing_times: tuple[tuple[int, float], ...]

    def processing_time(self, crane_count: int) -> float:
        try:
            return dict(self.processing_times)[int(crane_count)]
        except KeyError as exc:
            raise ValueError(
                f"vessel {self.vessel_index} has no processing time for {crane_count} cranes"
            ) from exc

    @property
    def vessel_id(self) -> str:
        return f"public_v{self.vessel_index:02d}"

    @property
    def cargo_class(self) -> str:
        return f"bacasp_s_v{self.vessel_index:02d}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "vessel_index": self.vessel_index,
            "vessel_id": self.vessel_id,
            "length": self.length,
            "arrival_time": self.arrival_time,
            "desired_departure": self.desired_departure,
            "position_cost": self.position_cost,
            "waiting_cost": self.waiting_cost,
            "delay_cost": self.delay_cost,
            "desired_position": self.desired_position,
            "min_cranes": self.min_cranes,
            "max_cranes": self.max_cranes,
            "processing_times": {str(key): value for key, value in self.processing_times},
        }


@dataclass(frozen=True)
class BacaspSRawInstance:
    source_path: str
    raw_sha256: str
    raw_byte_count: int
    raw_line_count: int
    quay_length: float
    horizon: float
    crane_count: int
    vessel_count: int
    crane_speed: float
    crane_setup_time: float
    vessels: tuple[BacaspSVesselRecord, ...]

    def snapshot(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "raw_sha256": self.raw_sha256,
            "raw_byte_count": self.raw_byte_count,
            "raw_line_count": self.raw_line_count,
            "quay_length": self.quay_length,
            "horizon": self.horizon,
            "crane_count": self.crane_count,
            "vessel_count": self.vessel_count,
            "crane_speed": self.crane_speed,
            "crane_setup_time": self.crane_setup_time,
            "vessels": [row.snapshot() for row in self.vessels],
        }


@dataclass(frozen=True)
class AdaptedPublicPortInstance:
    source: BacaspSRawInstance
    port_instance: PortInstance
    source_records: Mapping[str, BacaspSVesselRecord]
    berth_intervals: Mapping[str, tuple[float, float]]
    augmentation: Mapping[str, Any]

    def snapshot(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source.raw_sha256,
            "port_instance_name": self.port_instance.name,
            "port_instance_digest": self.port_instance.digest,
            "berth_intervals": {
                key: [float(value[0]), float(value[1])]
                for key, value in sorted(self.berth_intervals.items())
            },
            "augmentation": _canonical(self.augmentation),
        }


def load_source_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("public source manifest must be a JSON object")
    if raw.get("schema_version") != "scheduleurm.port_public_source_manifest.v1":
        raise ValueError("unsupported public source manifest schema")
    return raw


def verify_public_fixture(
    fixture_path: Path | str = DEFAULT_FIXTURE,
    manifest_path: Path | str = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    fixture = Path(fixture_path)
    manifest = load_source_manifest(manifest_path)
    expected = manifest["fixture"]
    payload = fixture.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    actual_lines = len(payload.splitlines())
    checks = {
        "sha256": actual_hash == str(expected["sha256"]),
        "byte_count": len(payload) == int(expected["byte_count"]),
        "line_count": actual_lines == int(expected["line_count"]),
        "raw_bytes_declared_unmodified": expected.get("raw_bytes_modified") is False,
        "pinned_commit": len(str(manifest["dataset"]["repository_commit"])) == 40,
        "redistribution_license": manifest["dataset"]["license"]["spdx"] == "CC-BY-SA-4.0",
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "actual_sha256": actual_hash,
        "actual_byte_count": len(payload),
        "actual_line_count": actual_lines,
        "expected_sha256": str(expected["sha256"]),
        "raw_url": str(expected["raw_url"]),
        "repository_commit": str(manifest["dataset"]["repository_commit"]),
        "license": _canonical(manifest["dataset"]["license"]),
    }


def parse_bacasp_s_instance(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
) -> BacaspSRawInstance:
    source = Path(path)
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"BACASP-S fixture SHA256 mismatch: {digest} != {expected_sha256}")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("BACASP-S fixture must be ASCII text") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("BACASP-S fixture is incomplete")

    header = _number_fields(lines[0], expected_count=4, label="header")
    quay_length, horizon = float(header[0]), float(header[1])
    crane_count = _integer_value(header[2], "crane_count")
    vessel_count = _integer_value(header[3], "vessel_count")
    crane_tokens = lines[1].split()
    if len(crane_tokens) != 3 or crane_tokens[0] != "C":
        raise ValueError("BACASP-S LargeMB fixture must contain a 'C speed setup' line")
    crane_speed = _positive_float(crane_tokens[1], "crane_speed")
    crane_setup = _nonnegative_float(crane_tokens[2], "crane_setup_time")
    if len(lines) != vessel_count + 2:
        raise ValueError(
            f"BACASP-S vessel row count mismatch: expected {vessel_count}, got {len(lines) - 2}"
        )
    if quay_length <= 0.0 or horizon <= 0.0 or crane_count <= 0 or vessel_count <= 0:
        raise ValueError("BACASP-S header values must be positive")

    vessels = tuple(
        _parse_vessel_line(line, index=index, quay_length=quay_length, crane_count=crane_count)
        for index, line in enumerate(lines[2:], start=1)
    )
    return BacaspSRawInstance(
        source_path=str(source),
        raw_sha256=digest,
        raw_byte_count=len(payload),
        raw_line_count=len(payload.splitlines()),
        quay_length=quay_length,
        horizon=horizon,
        crane_count=crane_count,
        vessel_count=vessel_count,
        crane_speed=crane_speed,
        crane_setup_time=crane_setup,
        vessels=vessels,
    )


def _parse_vessel_line(
    line: str,
    *,
    index: int,
    quay_length: float,
    crane_count: int,
) -> BacaspSVesselRecord:
    tokens = line.split()
    if len(tokens) < 10:
        raise ValueError(f"vessel row {index} is too short")
    values = [_finite_float(token, f"vessel[{index}]") for token in tokens]
    length, arrival, departure = values[0:3]
    position_cost, waiting_cost, delay_cost = values[3:6]
    desired_position = values[6]
    min_cranes = _integer_value(values[7], f"vessel[{index}].min_cranes")
    max_cranes = _integer_value(values[8], f"vessel[{index}].max_cranes")
    expected_count = 9 + max_cranes - min_cranes + 1
    if len(values) != expected_count:
        raise ValueError(
            f"vessel row {index} has {len(values)} fields; expected {expected_count} for "
            f"q=[{min_cranes},{max_cranes}]"
        )
    if not 0.0 < length <= quay_length:
        raise ValueError(f"vessel row {index} has invalid length")
    if arrival < 0.0 or departure < arrival:
        raise ValueError(f"vessel row {index} has invalid arrival/departure")
    if any(value < 0.0 for value in (position_cost, waiting_cost, delay_cost)):
        raise ValueError(f"vessel row {index} has negative cost")
    # The public files use one-based quay coordinates even though the paper's
    # model is written with a zero-based coordinate.  LargeMB therefore admits
    # b_i in [1, L + 1 - l_i].  Preserve the source value in the raw record and
    # convert only at the fixed-slot adapter boundary.
    if desired_position < 1.0 or desired_position + length > quay_length + 1.0 + EPS:
        raise ValueError(f"vessel row {index} has infeasible desired position")
    if min_cranes <= 0 or max_cranes < min_cranes or max_cranes > crane_count:
        raise ValueError(f"vessel row {index} has invalid crane bounds")
    processing = tuple(
        (cranes, _positive_float(values[9 + offset], f"vessel[{index}].u[{cranes}]"))
        for offset, cranes in enumerate(range(min_cranes, max_cranes + 1))
    )
    return BacaspSVesselRecord(
        vessel_index=index,
        length=length,
        arrival_time=arrival,
        desired_departure=departure,
        position_cost=position_cost,
        waiting_cost=waiting_cost,
        delay_cost=delay_cost,
        desired_position=desired_position,
        min_cranes=min_cranes,
        max_cranes=max_cranes,
        processing_times=processing,
    )


def adapt_bacasp_s_to_port_instance(source: BacaspSRawInstance) -> AdaptedPublicPortInstance:
    intervals = _fixed_berth_intervals(source)
    berths = tuple(
        BerthSpec(
            berth_id=berth_id,
            position=start,
            max_length=end - start,
            max_draft=1.0,
            rate_multiplier=1.0,
        )
        for berth_id, (start, end) in intervals.items()
    )
    compatible_berths = tuple(row.berth_id for row in berths)
    cargo_rates = tuple(
        sorted(
            (
                row.cargo_class,
                1.0 / row.processing_time(row.min_cranes),
            )
            for row in source.vessels
        )
    )
    cranes = tuple(
        QuayCraneSpec(
            crane_id=f"qc_{index:02d}",
            base_rate=1.0,
            compatible_berths=compatible_berths,
            cargo_multipliers=cargo_rates,
        )
        for index in range(1, source.crane_count + 1)
    )

    cargo_classes = tuple(row.cargo_class for row in source.vessels)
    yard_near_rates = tuple(
        (row.cargo_class, 1.0 / max(0.25, 0.12 * min(dict(row.processing_times).values())))
        for row in source.vessels
    )
    yard_far_rates = tuple((key, rate * 0.78) for key, rate in yard_near_rates)
    gate_fast_rates = tuple(
        (row.cargo_class, 1.0 / max(0.20, 0.08 * min(dict(row.processing_times).values())))
        for row in source.vessels
    )
    gate_regular_rates = tuple((key, rate * 0.82) for key, rate in gate_fast_rates)
    yards = (
        YardBlockSpec("synthetic_yard_near", 0.0, cargo_classes, yard_near_rates),
        YardBlockSpec("synthetic_yard_far", source.quay_length, cargo_classes, yard_far_rates),
    )
    gates = (
        GateLaneSpec("synthetic_gate_fast", cargo_classes, gate_fast_rates),
        GateLaneSpec("synthetic_gate_regular", cargo_classes, gate_regular_rates),
    )

    berth_map = {row.berth_id: row for row in berths}
    vessels = tuple(
        VesselSpec(
            vessel_id=row.vessel_id,
            arrival_time=row.arrival_time,
            due_time=row.desired_departure,
            priority_weight=max(1.0, row.waiting_cost / max(1.0, row.position_cost)),
            length=row.length,
            draft=1.0,
            cargo_class=row.cargo_class,
            quay_work=1.0,
            yard_work=1.0,
            gate_work=1.0,
            preferred_berth=_preferred_fixed_berth(row, berth_map),
            preferred_yard=(
                "synthetic_yard_near"
                if row.desired_position <= source.quay_length / 2.0
                else "synthetic_yard_far"
            ),
            max_cranes=row.max_cranes,
        )
        for row in source.vessels
    )
    instance = PortInstance(
        name="public_bacasp_s_large_mb_30_fixed_slot_projection",
        berths=berths,
        quay_cranes=cranes,
        yard_blocks=yards,
        gate_lanes=gates,
        vessels=vessels,
        lower_service_factor=1.0,
        decision_interval=1.0,
        reberth_base_duration=0.01,
        reberth_distance_duration=0.0,
        reberth_fixed_cost=0.0,
        crane_reassignment_base_duration=max(source.crane_setup_time, 0.01),
        crane_reassignment_distance_duration=1.0 / source.crane_speed,
        crane_reassignment_fixed_cost=0.0,
        yard_rehandle_base_duration=0.01,
        yard_rehandle_distance_duration=0.0,
        yard_rehandle_fixed_cost=0.0,
        cost_to_penalty=0.0,
        max_reberths_per_vessel=0,
        max_crane_reassignments_per_vessel=0,
        max_yard_rehandles_per_vessel=0,
    )
    instance.validate()
    return AdaptedPublicPortInstance(
        source=source,
        port_instance=instance,
        source_records={row.vessel_id: row for row in source.vessels},
        berth_intervals=intervals,
        augmentation={
            "continuous_quay_projection": {
                "kind": "fixed_non_overlapping_slots",
                "conservative_subset": True,
                "source_quay_length": source.quay_length,
                "slot_count": len(intervals),
                "source_field": False,
            },
            "neutral_draft": {"value": 1.0, "source_field": False},
            "yard": {
                "kind": "synthetic_augmentation",
                "source_field": False,
                "work_rule": "one normalized unit per vessel",
                "rate_rule": "deterministic fraction of source minimum quay processing time",
            },
            "gate": {
                "kind": "synthetic_augmentation",
                "source_field": False,
                "work_rule": "one normalized unit per vessel",
                "rate_rule": "deterministic fraction of source minimum quay processing time",
            },
            "source_crane_semantics": {
                "time_invariant_during_service": True,
                "minimum_cranes_enforced_by_adapter": True,
                "processing_time_u_iq_used_exactly": True,
                "consecutive_crane_setup_and_travel_used": True,
                "mid_service_reassignment_enabled": False,
            },
            "initial_crane_state": {
                "kind": "adapter_assumption",
                "source_field": False,
                "rule": "first use has zero transition because initial crane positions are absent",
            },
        },
    )


class PublicBacaspSPortSimulator(PortSchedulingSimulator):
    """Existing port simulator with BACASP-S parser constraints and setup time."""

    def __init__(self, adapted: AdaptedPublicPortInstance, policy: str):
        self.adapted = adapted
        self.crane_last_position: dict[str, float | None] = {
            row.crane_id: None for row in adapted.port_instance.quay_cranes
        }
        self.public_crane_transition_time = 0.0
        super().__init__(adapted.port_instance, policy)

    def _quay_rate(
        self,
        runtime,
        berth_id: str,
        crane_ids: Iterable[str],
        *,
        lower: bool,
    ) -> float:
        record = self.adapted.source_records[runtime.spec.vessel_id]
        crane_count = len(tuple(crane_ids))
        if crane_count < record.min_cranes or crane_count > record.max_cranes:
            return 0.0
        rate = 1.0 / record.processing_time(crane_count)
        return rate * self.instance.lower_service_factor if lower else rate

    def _crane_bundles(
        self,
        vessel: VesselSpec,
        berth_id: str,
        crane_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], ...]:
        """Enumerate every available consecutive specific-crane segment."""
        record = self.adapted.source_records[vessel.vessel_id]
        compatible = sorted(
            (
                crane_id
                for crane_id in crane_ids
                if berth_id in self.cranes[crane_id].compatible_berths
            ),
            key=_crane_number,
        )
        bundles: set[tuple[str, ...]] = set()
        for size in range(record.min_cranes, record.max_cranes + 1):
            for start in range(0, len(compatible) - size + 1):
                candidate = tuple(compatible[start : start + size])
                if _consecutive_cranes(candidate):
                    bundles.add(candidate)
        return tuple(sorted(bundles, key=lambda row: (len(row), row)))

    def _quay_start_actions(self) -> list[AtomicAction]:
        out: list[AtomicAction] = []
        for action in super()._quay_start_actions():
            record = self.adapted.source_records[action.vessel_id]
            cranes = tuple(str(value) for value in action.metadata["crane_ids"])
            if not record.min_cranes <= len(cranes) <= record.max_cranes:
                continue
            if self._crosses_active_assignment(action):
                continue
            berth_id = str(action.metadata["berth_id"])
            target_position = self._vessel_midpoint(record, berth_id)
            setup_by_crane: dict[str, float] = {}
            for crane_id in cranes:
                prior = self.crane_last_position[crane_id]
                setup_by_crane[crane_id] = (
                    0.0
                    if prior is None
                    else self.adapted.source.crane_setup_time
                    + abs(prior - target_position) / self.adapted.source.crane_speed
                )
            setup_duration = max(setup_by_crane.values(), default=0.0)
            source_processing_time = record.processing_time(len(cranes))
            berth_start = self.adapted.berth_intervals[berth_id][0]
            mapped_source_position = berth_start + 1.0
            position_deviation = abs(mapped_source_position - record.desired_position)
            normalized_position_penalty = (
                position_deviation / max(1.0, self.adapted.source.quay_length)
            )
            metadata = dict(action.metadata)
            metadata.update(
                {
                    "source_min_cranes": record.min_cranes,
                    "source_max_cranes": record.max_cranes,
                    "source_processing_time": source_processing_time,
                    "source_desired_position": record.desired_position,
                    "mapped_berth_position": mapped_source_position,
                    "position_deviation": position_deviation,
                    "crane_setup_travel_duration": setup_duration,
                    "crane_setup_travel_by_crane": setup_by_crane,
                    "crane_speed": self.adapted.source.crane_speed,
                    "crane_setup_time": self.adapted.source.crane_setup_time,
                }
            )
            out.append(
                replace(
                    action,
                    penalty_units=(
                        action.penalty_units + setup_duration + normalized_position_penalty
                    ),
                    estimated_duration=source_processing_time + setup_duration,
                    metadata=metadata,
                )
            )
        return out

    def feasible_atomic_actions(self) -> tuple[AtomicAction, ...]:
        actions = list(super().feasible_atomic_actions())
        quay_indices = [
            index for index, row in enumerate(actions) if row.action_type == "start_quay"
        ]
        additions: dict[int, list[str]] = {index: [] for index in quay_indices}
        for left_pos, left_index in enumerate(quay_indices):
            for right_index in quay_indices[left_pos + 1 :]:
                left, right = actions[left_index], actions[right_index]
                if left.vessel_id == right.vessel_id:
                    continue
                if self._assignments_cross(left, right):
                    token = "rail_conflict:" + hashlib.sha256(
                        "|".join(sorted((left.action_id, right.action_id))).encode("utf-8")
                    ).hexdigest()[:16]
                    additions[left_index].append(token)
                    additions[right_index].append(token)
        for index, tokens in additions.items():
            if tokens:
                action = actions[index]
                actions[index] = replace(
                    action,
                    conflict_tokens=tuple(sorted(set(action.conflict_tokens) | set(tokens))),
                )
        return tuple(sorted(actions, key=lambda row: row.action_id))

    def check_action_feasible(self, action: AtomicAction) -> tuple[bool, str]:
        ok, reason = super().check_action_feasible(action)
        if not ok or action.action_type != "start_quay":
            return ok, reason
        record = self.adapted.source_records[action.vessel_id]
        cranes = tuple(str(value) for value in action.metadata["crane_ids"])
        if not record.min_cranes <= len(cranes) <= record.max_cranes:
            return False, "source_crane_count_out_of_bounds"
        if not _consecutive_cranes(cranes):
            return False, "source_specific_cranes_not_consecutive"
        if self._crosses_active_assignment(action):
            return False, "source_crane_rail_order_crossing"
        return True, ""

    def _apply_action(self, action: AtomicAction) -> None:
        if action.action_type != "start_quay":
            super()._apply_action(action)
            return
        setup_duration = float(action.metadata.get("crane_setup_travel_duration") or 0.0)
        if setup_duration <= EPS:
            super()._apply_action(action)
            return
        runtime = self.vessels[action.vessel_id]
        berth_id = str(action.metadata["berth_id"])
        cranes = [str(value) for value in action.metadata["crane_ids"]]
        marker = f"transition:{runtime.spec.vessel_id}:public_quay_setup"
        self.action_counts[action.action_type] += 1
        runtime.stage = "reberthing"
        runtime.berth_id = None
        runtime.crane_ids = []
        self.berth_occupancy[berth_id] = marker
        for crane_id in cranes:
            self.crane_assignment[crane_id] = marker
        self.public_crane_transition_time += setup_duration
        self.transitions.append(
            PendingTransition(
                kind="public_quay_setup",
                vessel_id=runtime.spec.vessel_id,
                complete_at=self.clock + setup_duration,
                metadata={"berth_id": berth_id, "crane_ids": cranes},
            )
        )

    def _complete_transition(self, transition: PendingTransition) -> None:
        if transition.kind != "public_quay_setup":
            super()._complete_transition(transition)
            return
        runtime = self.vessels[transition.vessel_id]
        berth_id = str(transition.metadata["berth_id"])
        cranes = [str(value) for value in transition.metadata["crane_ids"]]
        runtime.stage = "quay"
        runtime.berth_id = berth_id
        runtime.crane_ids = cranes
        runtime.quay_start = self.clock if runtime.quay_start is None else runtime.quay_start
        self.berth_occupancy[berth_id] = runtime.spec.vessel_id
        for crane_id in cranes:
            self.crane_assignment[crane_id] = runtime.spec.vessel_id
        self._record_event(
            "transition_complete",
            kind="public_quay_setup",
            vessel_id=runtime.spec.vessel_id,
        )

    def _complete_quay(self, runtime) -> None:
        if runtime.berth_id:
            record = self.adapted.source_records[runtime.spec.vessel_id]
            midpoint = self._vessel_midpoint(record, runtime.berth_id)
            for crane_id in runtime.crane_ids:
                self.crane_last_position[crane_id] = midpoint
        super()._complete_quay(runtime)

    def _result(self) -> dict[str, Any]:
        result = super()._result()
        source_metrics, audit = self._source_core_metrics(result)
        result["public_source_core_metrics"] = source_metrics
        result["public_source_feasibility_audit"] = audit
        result["public_source_mapping"] = self.adapted.snapshot()
        result["public_source_claim_boundary"] = {
            "source_core_only": (
                "arrival, due, length, desired position, crane bounds, u_iq, alpha, beta"
            ),
            "synthetic_tail_stages": ["yard", "gate"],
            "initial_crane_position_observed": False,
            "mid_service_reconfiguration_admitted": False,
            "continuous_quay_reproduced": False,
            "published_objective_reproduced": False,
        }
        return result

    def _source_core_metrics(
        self,
        result: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        starts: dict[str, Mapping[str, Any]] = {}
        for decision in result["decision_audits"]:
            for action in decision["selected_actions"]:
                if action["action_type"] == "start_quay":
                    starts[str(action["task_id"])] = action
        quay_completion = {
            str(row["vessel_id"]): float(row["time"])
            for row in result["event_trace"]
            if row["event"] == "stage_complete" and row.get("stage") == "quay"
        }
        objective = 0.0
        flows: list[float] = []
        q_bounds_ready = True
        consecutive_ready = True
        processing_ready = True
        intervals: list[dict[str, Any]] = []
        for vessel_id, record in sorted(self.adapted.source_records.items()):
            runtime = self.vessels[vessel_id]
            action = starts[vessel_id]
            cranes = tuple(str(value) for value in action["metadata"]["crane_ids"])
            crane_count = len(cranes)
            q_bounds_ready = (
                q_bounds_ready and record.min_cranes <= crane_count <= record.max_cranes
            )
            consecutive_ready = consecutive_ready and _consecutive_cranes(cranes)
            complete = quay_completion[vessel_id]
            start = float(runtime.quay_start)
            actual_processing = complete - start
            expected_processing = record.processing_time(crane_count)
            processing_ready = (
                processing_ready and abs(actual_processing - expected_processing) <= 1e-6
            )
            wait = max(0.0, start - record.arrival_time)
            delay = max(0.0, complete - record.desired_departure)
            deviation = float(action["metadata"]["position_deviation"])
            objective += (
                record.waiting_cost * wait
                + record.delay_cost * delay
                + record.position_cost * deviation
            )
            flows.append(complete - record.arrival_time)
            intervals.append(
                {
                    "vessel_id": vessel_id,
                    "start": start,
                    "complete": complete,
                    "berth_id": action["metadata"]["berth_id"],
                    "crane_ids": list(cranes),
                }
            )
        rail_ready = _rail_endpoint_order_ready(intervals, self.adapted.berth_intervals)
        first_use_assumption_ready = self._first_crane_use_has_zero_transition(result)
        source_metrics = {
            "quay_makespan": _round(
                max(quay_completion.values())
                - min(row.arrival_time for row in self.adapted.source.vessels)
            ),
            "mean_quay_flow_time": _round(sum(flows) / len(flows)),
            "source_objective_cost": _round(objective),
            "crane_setup_travel_duration": _round(self.public_crane_transition_time),
            "completed_source_vessels": len(flows),
        }
        audit = {
            "source_min_max_cranes_ready": q_bounds_ready,
            "source_specific_cranes_consecutive_ready": consecutive_ready,
            "source_u_iq_processing_time_ready": processing_ready,
            "rail_endpoint_order_ready": rail_ready,
            "initial_crane_position_observed": False,
            "zero_transition_on_first_crane_use_ready": first_use_assumption_ready,
            "continuous_crane_trajectory_ready": False,
            "resource_feasibility_ready": all(
                (
                    q_bounds_ready,
                    consecutive_ready,
                    processing_ready,
                    rail_ready,
                    first_use_assumption_ready,
                )
            ),
        }
        return source_metrics, audit

    def _first_crane_use_has_zero_transition(self, result: Mapping[str, Any]) -> bool:
        seen: set[str] = set()
        for decision in result["decision_audits"]:
            for action in decision["selected_actions"]:
                if action["action_type"] != "start_quay":
                    continue
                cranes = [str(value) for value in action["metadata"]["crane_ids"]]
                by_crane = action["metadata"]["crane_setup_travel_by_crane"]
                for crane in cranes:
                    if crane not in seen and float(by_crane[crane]) > EPS:
                        return False
                seen.update(cranes)
        return bool(seen)

    def _crosses_active_assignment(self, action: AtomicAction) -> bool:
        berth_id = str(action.metadata["berth_id"])
        cranes = tuple(str(value) for value in action.metadata["crane_ids"])
        new_position = self.adapted.berth_intervals[berth_id][0]
        for runtime in self.vessels.values():
            if runtime.stage == "quay" and runtime.berth_id and runtime.crane_ids:
                active_position = self.adapted.berth_intervals[runtime.berth_id][0]
                if _crane_orders_cross(new_position, cranes, active_position, runtime.crane_ids):
                    return True
        for transition in self.transitions:
            if transition.kind != "public_quay_setup":
                continue
            active_berth = str(transition.metadata["berth_id"])
            active_position = self.adapted.berth_intervals[active_berth][0]
            active_cranes = tuple(str(value) for value in transition.metadata["crane_ids"])
            if _crane_orders_cross(new_position, cranes, active_position, active_cranes):
                return True
        return False

    def _assignments_cross(self, left: AtomicAction, right: AtomicAction) -> bool:
        left_berth = str(left.metadata["berth_id"])
        right_berth = str(right.metadata["berth_id"])
        return _crane_orders_cross(
            self.adapted.berth_intervals[left_berth][0],
            tuple(str(value) for value in left.metadata["crane_ids"]),
            self.adapted.berth_intervals[right_berth][0],
            tuple(str(value) for value in right.metadata["crane_ids"]),
        )

    def _vessel_midpoint(self, record: BacaspSVesselRecord, berth_id: str) -> float:
        return self.adapted.berth_intervals[berth_id][0] + record.length / 2.0


def run_public_port_policy(adapted: AdaptedPublicPortInstance, policy: str) -> dict[str, Any]:
    return PublicBacaspSPortSimulator(adapted, policy).run(max_events=40_000)


def build_port_public_benchmark(
    fixture_path: Path | str = DEFAULT_FIXTURE,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    *,
    policies: Sequence[str] = ALL_POLICIES,
    determinism_repeats: int = 2,
) -> dict[str, Any]:
    if determinism_repeats < 2:
        raise ValueError("determinism_repeats must be at least two")
    selected_policies = tuple(policies)
    if tuple(selected_policies) != tuple(ALL_POLICIES):
        raise ValueError("public benchmark must run the registered five-policy matrix")
    manifest = load_source_manifest(manifest_path)
    authenticity = verify_public_fixture(fixture_path, manifest_path)
    source = parse_bacasp_s_instance(
        fixture_path,
        expected_sha256=str(manifest["fixture"]["sha256"]),
    )
    adapted = adapt_bacasp_s_to_port_instance(source)

    results: dict[str, Any] = {}
    determinism: dict[str, Any] = {}
    for policy in selected_policies:
        repeats = [run_public_port_policy(adapted, policy) for _ in range(determinism_repeats)]
        hashes = [_result_digest(row) for row in repeats]
        results[policy] = repeats[0]
        determinism[policy] = {
            "repeat_count": determinism_repeats,
            "hashes": hashes,
            "deterministic": len(set(hashes)) == 1,
        }

    metric_schemas = {tuple(sorted(row["metrics"])) for row in results.values()}
    source_metric_schemas = {
        tuple(sorted(row["public_source_core_metrics"])) for row in results.values()
    }
    source_feasibility = all(
        bool(row["public_source_feasibility_audit"]["resource_feasibility_ready"])
        for row in results.values()
    )
    deterministic = all(row["deterministic"] for row in determinism.values())
    runs_ready = all(bool(row["pass"]) for row in results.values())
    no_mid_service_reconfiguration = all(
        all(
            int(row["action_counts"].get(action_type, 0)) == 0
            for action_type in ("reberth", "crane_reassignment", "yard_rehandle")
        )
        for row in results.values()
    )
    public_metrics = {
        policy: dict(row["public_source_core_metrics"])
        for policy, row in results.items()
    }
    pareto = _source_pareto(public_metrics)
    adapter_ready = all(
        (
            authenticity["ready"],
            source.vessel_count == len(source.vessels),
            source.vessel_count == 30,
            runs_ready,
            deterministic,
            len(metric_schemas) == 1,
            len(source_metric_schemas) == 1,
            source_feasibility,
            no_mid_service_reconfiguration,
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": "port_public_bacasp_s_adapter",
        "status": (
            "PORT_PUBLIC_BACASP_S_ADAPTER_PASS"
            if adapter_ready
            else "PORT_PUBLIC_BACASP_S_ADAPTER_OPEN"
        ),
        "pass": adapter_ready,
        "scoped_claim_ready": adapter_ready,
        "source_authenticity_ready": authenticity["ready"],
        "parser_ready": source.vessel_count == len(source.vessels) == 30,
        "same_policy_matrix_ready": tuple(results) == tuple(ALL_POLICIES),
        "same_metric_schema_ready": len(metric_schemas) == len(source_metric_schemas) == 1,
        "deterministic_reproduction_ready": deterministic,
        "resource_feasibility_ready": source_feasibility,
        "source_time_invariant_assignment_ready": no_mid_service_reconfiguration,
        "mid_service_reconfiguration_admitted": False,
        "continuous_quay_reproduction_ready": False,
        "physical_port_claim_ready": False,
        "published_optimum_comparison_ready": False,
        "source_authenticity": authenticity,
        "source_manifest": _canonical(manifest),
        "parsed_source": source.snapshot(),
        "adapted_instance": adapted.snapshot(),
        "policy_results": results,
        "determinism": determinism,
        "source_core_metrics": public_metrics,
        "source_core_pareto": pareto,
        "ours_source_core_pareto_nondominated": THEOREM_POLICY in pareto["nondominated_policies"],
        "claim_boundary": {
            "supports": [
                "byte-verified parsing of the pinned public BACASP-S LargeMB fixture",
                "source q_min/q_max and u_iq processing-time feasibility in the mapped run",
                "source crane speed/setup parameters in event-time crane transitions",
                (
                    "same-policy comparison of robust MaxWeight, FCFS, SPT, EDD, "
                    "and reconfiguration-aware greedy under the source time-invariant "
                    "action family"
                ),
                "fixed-slot and rail-endpoint resource-feasibility audit",
            ],
            "does_not_support": [
                "reproduction of the paper's continuous-quay MILP or published bounds",
                (
                    "equivalence between the fixed-slot candidate family and the "
                    "unrestricted continuous quay"
                ),
                "continuous crane-path collision verification between assignment endpoints",
                "a claim that BACASP-S contains yard, gate, or vessel-draft observations",
                (
                    "evidence for mid-service re-berthing, crane reassignment, or yard "
                    "rehandle on this source instance"
                ),
                (
                    "physical-port performance, safety, or superiority over the authors' "
                    "exact algorithm"
                ),
            ],
            "augmentation_wording": (
                "Yard, gate, neutral draft, fixed berth slots, and the zero-transition "
                "initial crane state are Scheduleurm schema augmentations; "
                "all source-core metrics stop at quay completion."
            ),
        },
    }
    report["artifact_hash"] = _digest(
        {key: value for key, value in report.items() if key != "artifact_hash"}
    )
    return report


def _fixed_berth_intervals(source: BacaspSRawInstance) -> dict[str, tuple[float, float]]:
    max_length = max(row.length for row in source.vessels)
    min_length = min(row.length for row in source.vessels)
    remaining = source.quay_length
    start = 0.0
    widths: list[float] = []
    while remaining > EPS:
        width = min(max_length, remaining)
        widths.append(width)
        remaining -= width
    if len(widths) > 1 and widths[-1] < min_length:
        widths[-2] += widths[-1]
        widths.pop()
    intervals: dict[str, tuple[float, float]] = {}
    for index, width in enumerate(widths, start=1):
        intervals[f"public_berth_{index:02d}"] = (start, start + width)
        start += width
    if abs(start - source.quay_length) > EPS:
        raise ValueError("fixed berth projection does not cover the source quay length")
    if not any(end - begin >= max_length for begin, end in intervals.values()):
        raise ValueError("fixed berth projection cannot accommodate the largest vessel")
    return intervals


def _preferred_fixed_berth(
    vessel: BacaspSVesselRecord,
    berths: Mapping[str, BerthSpec],
) -> str:
    feasible = [row for row in berths.values() if vessel.length <= row.max_length + EPS]
    if not feasible:
        raise ValueError(f"no fixed berth can accommodate public vessel {vessel.vessel_index}")
    return min(
        feasible,
        key=lambda row: (abs((row.position + 1.0) - vessel.desired_position), row.berth_id),
    ).berth_id


def _rail_endpoint_order_ready(
    intervals: Sequence[Mapping[str, Any]],
    berth_intervals: Mapping[str, tuple[float, float]],
) -> bool:
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            if min(float(left["complete"]), float(right["complete"])) <= max(
                float(left["start"]), float(right["start"])
            ) + EPS:
                continue
            left_position = berth_intervals[str(left["berth_id"])][0]
            right_position = berth_intervals[str(right["berth_id"])][0]
            if _crane_orders_cross(
                left_position,
                tuple(str(value) for value in left["crane_ids"]),
                right_position,
                tuple(str(value) for value in right["crane_ids"]),
            ):
                return False
    return True


def _crane_orders_cross(
    left_position: float,
    left_cranes: Sequence[str],
    right_position: float,
    right_cranes: Sequence[str],
) -> bool:
    if abs(left_position - right_position) <= EPS:
        return True
    left_numbers = [_crane_number(value) for value in left_cranes]
    right_numbers = [_crane_number(value) for value in right_cranes]
    if left_position < right_position:
        return max(left_numbers) >= min(right_numbers)
    return max(right_numbers) >= min(left_numbers)


def _consecutive_cranes(cranes: Sequence[str]) -> bool:
    values = sorted(_crane_number(value) for value in cranes)
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def _crane_number(value: str) -> int:
    try:
        return int(str(value).rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"invalid public crane identifier: {value}") from exc


def _source_pareto(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    dominated_by: dict[str, list[str]] = {}
    nondominated: list[str] = []
    for policy, row in metrics.items():
        dominators = [
            other
            for other, other_row in metrics.items()
            if other != policy and _dominates_source(other_row, row)
        ]
        dominated_by[policy] = sorted(dominators)
        if not dominators:
            nondominated.append(policy)
    return {
        "metrics": list(SOURCE_COST_METRICS),
        "nondominated_policies": sorted(nondominated),
        "dominated_by": dominated_by,
    }


def _dominates_source(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = all(float(left[key]) <= float(right[key]) + 1e-8 for key in SOURCE_COST_METRICS)
    strict = any(float(left[key]) < float(right[key]) - 1e-8 for key in SOURCE_COST_METRICS)
    return no_worse and strict


def _result_digest(result: Mapping[str, Any]) -> str:
    return _digest(
        {
            "result_hash": result["result_hash"],
            "source_core_metrics": result["public_source_core_metrics"],
            "source_feasibility": result["public_source_feasibility_audit"],
            "mapping": result["public_source_mapping"],
        }
    )


def _number_fields(line: str, *, expected_count: int, label: str) -> list[float]:
    tokens = line.split()
    if len(tokens) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} fields")
    return [_finite_float(token, label) for token in tokens]


def _finite_float(value: Any, label: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(output):
        raise ValueError(f"{label} must be finite")
    return output


def _integer_value(value: Any, label: str) -> int:
    output = _finite_float(value, label)
    rounded = round(output)
    if abs(output - rounded) > EPS:
        raise ValueError(f"{label} must be an integer")
    return int(rounded)


def _positive_float(value: Any, label: str) -> float:
    output = _finite_float(value, label)
    if output <= 0.0:
        raise ValueError(f"{label} must be positive")
    return output


def _nonnegative_float(value: Any, label: str) -> float:
    output = _finite_float(value, label)
    if output < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return output


def _round(value: float) -> float:
    return round(float(value), 9)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        return _round(value)
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--determinism-repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_port_public_benchmark(
        args.fixture,
        args.manifest,
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
    "AdaptedPublicPortInstance",
    "BacaspSRawInstance",
    "BacaspSVesselRecord",
    "DEFAULT_FIXTURE",
    "DEFAULT_MANIFEST",
    "PublicBacaspSPortSimulator",
    "adapt_bacasp_s_to_port_instance",
    "build_port_public_benchmark",
    "load_source_manifest",
    "parse_bacasp_s_instance",
    "run_public_port_policy",
    "verify_public_fixture",
]
