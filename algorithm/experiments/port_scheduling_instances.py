"""Deterministic heterogeneous port instances for the OR sidecar benchmark.

The instances are deliberately finite and fully registered.  Their rates are
synthetic lower-service inputs, not measurements from a physical terminal.
Keeping the instance contract separate from the simulator makes the resulting
action ledger hashable and prevents a benchmark policy from changing the data
it is evaluated on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class BerthSpec:
    berth_id: str
    position: float
    max_length: float
    max_draft: float
    rate_multiplier: float


@dataclass(frozen=True)
class QuayCraneSpec:
    crane_id: str
    base_rate: float
    compatible_berths: tuple[str, ...]
    cargo_multipliers: tuple[tuple[str, float], ...] = ()

    def cargo_multiplier(self, cargo_class: str) -> float:
        return dict(self.cargo_multipliers).get(cargo_class, 1.0)


@dataclass(frozen=True)
class YardBlockSpec:
    block_id: str
    position: float
    compatible_cargo: tuple[str, ...]
    handling_rates: tuple[tuple[str, float], ...]

    def rate(self, cargo_class: str) -> float:
        return dict(self.handling_rates).get(cargo_class, 0.0)


@dataclass(frozen=True)
class GateLaneSpec:
    lane_id: str
    compatible_cargo: tuple[str, ...]
    processing_rates: tuple[tuple[str, float], ...]

    def rate(self, cargo_class: str) -> float:
        return dict(self.processing_rates).get(cargo_class, 0.0)


@dataclass(frozen=True)
class VesselSpec:
    vessel_id: str
    arrival_time: float
    due_time: float
    priority_weight: float
    length: float
    draft: float
    cargo_class: str
    quay_work: float
    yard_work: float
    gate_work: float
    preferred_berth: str
    preferred_yard: str
    max_cranes: int = 2


@dataclass(frozen=True)
class PortInstance:
    name: str
    berths: tuple[BerthSpec, ...]
    quay_cranes: tuple[QuayCraneSpec, ...]
    yard_blocks: tuple[YardBlockSpec, ...]
    gate_lanes: tuple[GateLaneSpec, ...]
    vessels: tuple[VesselSpec, ...]
    lower_service_factor: float = 0.90
    decision_interval: float = 8.0
    reberth_base_duration: float = 8.0
    reberth_distance_duration: float = 2.0
    reberth_fixed_cost: float = 5.0
    crane_reassignment_base_duration: float = 3.0
    crane_reassignment_distance_duration: float = 0.8
    crane_reassignment_fixed_cost: float = 1.5
    yard_rehandle_base_duration: float = 5.0
    yard_rehandle_distance_duration: float = 1.2
    yard_rehandle_fixed_cost: float = 3.0
    cost_to_penalty: float = 0.25
    max_reberths_per_vessel: int = 1
    max_crane_reassignments_per_vessel: int = 2
    max_yard_rehandles_per_vessel: int = 1

    def validate(self) -> None:
        if not self.name:
            raise ValueError("port instance name is required")
        if not self.berths or not self.quay_cranes or not self.yard_blocks or not self.gate_lanes:
            raise ValueError("all four port resource families must be non-empty")
        if not self.vessels:
            raise ValueError("a port instance must contain at least one vessel")
        if not 0.0 < self.lower_service_factor <= 1.0:
            raise ValueError("lower_service_factor must be in (0, 1]")
        if self.decision_interval <= 0.0:
            raise ValueError("decision_interval must be positive")

        berth_ids = _unique_ids("berth", (row.berth_id for row in self.berths))
        crane_ids = _unique_ids("quay crane", (row.crane_id for row in self.quay_cranes))
        yard_ids = _unique_ids("yard block", (row.block_id for row in self.yard_blocks))
        gate_ids = _unique_ids("gate lane", (row.lane_id for row in self.gate_lanes))
        _unique_ids("vessel", (row.vessel_id for row in self.vessels))

        for berth in self.berths:
            _require_positive(berth.max_length, f"{berth.berth_id}.max_length")
            _require_positive(berth.max_draft, f"{berth.berth_id}.max_draft")
            _require_positive(berth.rate_multiplier, f"{berth.berth_id}.rate_multiplier")
        for crane in self.quay_cranes:
            _require_positive(crane.base_rate, f"{crane.crane_id}.base_rate")
            if not crane.compatible_berths or not set(crane.compatible_berths) <= berth_ids:
                raise ValueError(f"{crane.crane_id} has invalid compatible_berths")
            for cargo, value in crane.cargo_multipliers:
                if not cargo or value <= 0.0:
                    raise ValueError(f"{crane.crane_id} has invalid cargo multiplier")
        for block in self.yard_blocks:
            if not block.compatible_cargo:
                raise ValueError(f"{block.block_id} has no compatible cargo")
            for cargo, value in block.handling_rates:
                if cargo not in block.compatible_cargo or value <= 0.0:
                    raise ValueError(f"{block.block_id} has invalid handling rate")
        for lane in self.gate_lanes:
            if not lane.compatible_cargo:
                raise ValueError(f"{lane.lane_id} has no compatible cargo")
            for cargo, value in lane.processing_rates:
                if cargo not in lane.compatible_cargo or value <= 0.0:
                    raise ValueError(f"{lane.lane_id} has invalid processing rate")

        berth_map = {row.berth_id: row for row in self.berths}
        for vessel in self.vessels:
            if vessel.arrival_time < 0.0 or vessel.due_time < vessel.arrival_time:
                raise ValueError(f"{vessel.vessel_id} has invalid arrival/due time")
            if vessel.priority_weight <= 0.0 or vessel.max_cranes <= 0:
                raise ValueError(f"{vessel.vessel_id} has invalid priority/max_cranes")
            for field, value in (
                ("length", vessel.length),
                ("draft", vessel.draft),
                ("quay_work", vessel.quay_work),
                ("yard_work", vessel.yard_work),
                ("gate_work", vessel.gate_work),
            ):
                _require_positive(value, f"{vessel.vessel_id}.{field}")
            if vessel.preferred_berth not in berth_ids or vessel.preferred_yard not in yard_ids:
                raise ValueError(f"{vessel.vessel_id} references an unknown preferred resource")
            feasible_berths = [
                berth.berth_id
                for berth in self.berths
                if vessel.length <= berth.max_length and vessel.draft <= berth.max_draft
            ]
            if not feasible_berths:
                raise ValueError(f"{vessel.vessel_id} has no feasible berth")
            if not any(set(crane.compatible_berths) & set(feasible_berths) for crane in self.quay_cranes):
                raise ValueError(f"{vessel.vessel_id} has no feasible quay crane")
            if not any(block.rate(vessel.cargo_class) > 0.0 for block in self.yard_blocks):
                raise ValueError(f"{vessel.vessel_id} has no feasible yard block")
            if not any(lane.rate(vessel.cargo_class) > 0.0 for lane in self.gate_lanes):
                raise ValueError(f"{vessel.vessel_id} has no feasible gate lane")
            preferred = berth_map[vessel.preferred_berth]
            if vessel.length > preferred.max_length or vessel.draft > preferred.max_draft:
                raise ValueError(f"{vessel.vessel_id} does not fit its preferred berth")

        if not crane_ids or not gate_ids:
            raise ValueError("resource identifiers are incomplete")
        for value, label in (
            (self.reberth_base_duration, "reberth_base_duration"),
            (self.crane_reassignment_base_duration, "crane_reassignment_base_duration"),
            (self.yard_rehandle_base_duration, "yard_rehandle_base_duration"),
        ):
            _require_positive(value, label)
        for value, label in (
            (self.reberth_distance_duration, "reberth_distance_duration"),
            (self.reberth_fixed_cost, "reberth_fixed_cost"),
            (self.crane_reassignment_distance_duration, "crane_reassignment_distance_duration"),
            (self.crane_reassignment_fixed_cost, "crane_reassignment_fixed_cost"),
            (self.yard_rehandle_distance_duration, "yard_rehandle_distance_duration"),
            (self.yard_rehandle_fixed_cost, "yard_rehandle_fixed_cost"),
            (self.cost_to_penalty, "cost_to_penalty"),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{label} must be nonnegative")
        for value, label in (
            (self.max_reberths_per_vessel, "max_reberths_per_vessel"),
            (self.max_crane_reassignments_per_vessel, "max_crane_reassignments_per_vessel"),
            (self.max_yard_rehandles_per_vessel, "max_yard_rehandles_per_vessel"),
        ):
            if int(value) < 0:
                raise ValueError(f"{label} must be nonnegative")

    def snapshot(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "berths": [asdict(row) for row in self.berths],
            "quay_cranes": [asdict(row) for row in self.quay_cranes],
            "yard_blocks": [asdict(row) for row in self.yard_blocks],
            "gate_lanes": [asdict(row) for row in self.gate_lanes],
            "vessels": [asdict(row) for row in self.vessels],
            "lower_service_factor": self.lower_service_factor,
            "decision_interval": self.decision_interval,
            "reberth_base_duration": self.reberth_base_duration,
            "reberth_distance_duration": self.reberth_distance_duration,
            "reberth_fixed_cost": self.reberth_fixed_cost,
            "crane_reassignment_base_duration": self.crane_reassignment_base_duration,
            "crane_reassignment_distance_duration": self.crane_reassignment_distance_duration,
            "crane_reassignment_fixed_cost": self.crane_reassignment_fixed_cost,
            "yard_rehandle_base_duration": self.yard_rehandle_base_duration,
            "yard_rehandle_distance_duration": self.yard_rehandle_distance_duration,
            "yard_rehandle_fixed_cost": self.yard_rehandle_fixed_cost,
            "cost_to_penalty": self.cost_to_penalty,
            "max_reberths_per_vessel": self.max_reberths_per_vessel,
            "max_crane_reassignments_per_vessel": self.max_crane_reassignments_per_vessel,
            "max_yard_rehandles_per_vessel": self.max_yard_rehandles_per_vessel,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_port_instance(path: Path | str) -> PortInstance:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("port instance JSON must be an object")
    instance = port_instance_from_mapping(raw)
    instance.validate()
    return instance


def port_instance_from_mapping(raw: Mapping[str, Any]) -> PortInstance:
    def pairs(value: Any) -> tuple[tuple[str, float], ...]:
        if isinstance(value, Mapping):
            rows = value.items()
        elif isinstance(value, (list, tuple)):
            rows = value
        else:
            raise ValueError("rate maps must be JSON objects or key-rate pairs")
        try:
            return tuple(sorted((str(key), float(rate)) for key, rate in rows))
        except (TypeError, ValueError) as exc:
            raise ValueError("rate maps must contain key-rate pairs") from exc

    return PortInstance(
        name=str(raw["name"]),
        berths=tuple(
            BerthSpec(
                berth_id=str(row["berth_id"]),
                position=float(row["position"]),
                max_length=float(row["max_length"]),
                max_draft=float(row["max_draft"]),
                rate_multiplier=float(row["rate_multiplier"]),
            )
            for row in raw["berths"]
        ),
        quay_cranes=tuple(
            QuayCraneSpec(
                crane_id=str(row["crane_id"]),
                base_rate=float(row["base_rate"]),
                compatible_berths=tuple(str(item) for item in row["compatible_berths"]),
                cargo_multipliers=pairs(row.get("cargo_multipliers", {})),
            )
            for row in raw["quay_cranes"]
        ),
        yard_blocks=tuple(
            YardBlockSpec(
                block_id=str(row["block_id"]),
                position=float(row["position"]),
                compatible_cargo=tuple(str(item) for item in row["compatible_cargo"]),
                handling_rates=pairs(row["handling_rates"]),
            )
            for row in raw["yard_blocks"]
        ),
        gate_lanes=tuple(
            GateLaneSpec(
                lane_id=str(row["lane_id"]),
                compatible_cargo=tuple(str(item) for item in row["compatible_cargo"]),
                processing_rates=pairs(row["processing_rates"]),
            )
            for row in raw["gate_lanes"]
        ),
        vessels=tuple(
            VesselSpec(
                vessel_id=str(row["vessel_id"]),
                arrival_time=float(row["arrival_time"]),
                due_time=float(row["due_time"]),
                priority_weight=float(row["priority_weight"]),
                length=float(row["length"]),
                draft=float(row["draft"]),
                cargo_class=str(row["cargo_class"]),
                quay_work=float(row["quay_work"]),
                yard_work=float(row["yard_work"]),
                gate_work=float(row["gate_work"]),
                preferred_berth=str(row["preferred_berth"]),
                preferred_yard=str(row["preferred_yard"]),
                max_cranes=int(row.get("max_cranes", 2)),
            )
            for row in raw["vessels"]
        ),
        **{
            key: raw[key]
            for key in (
                "lower_service_factor",
                "decision_interval",
                "reberth_base_duration",
                "reberth_distance_duration",
                "reberth_fixed_cost",
                "crane_reassignment_base_duration",
                "crane_reassignment_distance_duration",
                "crane_reassignment_fixed_cost",
                "yard_rehandle_base_duration",
                "yard_rehandle_distance_duration",
                "yard_rehandle_fixed_cost",
                "cost_to_penalty",
                "max_reberths_per_vessel",
                "max_crane_reassignments_per_vessel",
                "max_yard_rehandles_per_vessel",
            )
            if key in raw
        },
    )


def built_in_port_instances() -> tuple[PortInstance, ...]:
    instances = (
        _balanced_instance(),
        _berth_crane_pressure_instance(),
        _yard_gate_surge_instance(),
    )
    for instance in instances:
        instance.validate()
    return instances


def _resource_bundle() -> dict[str, tuple[Any, ...]]:
    berths = (
        BerthSpec("b_deep", 0.0, 360.0, 15.0, 1.18),
        BerthSpec("b_mid", 1.0, 285.0, 12.5, 1.00),
        BerthSpec("b_feeder", 2.0, 220.0, 10.0, 0.84),
    )
    cranes = (
        QuayCraneSpec("qc_fast", 2.20, ("b_deep", "b_mid"), (("reefer", 0.92),)),
        QuayCraneSpec("qc_a", 1.75, ("b_deep", "b_mid", "b_feeder")),
        QuayCraneSpec("qc_b", 1.55, ("b_deep", "b_mid", "b_feeder"), (("bulk", 1.08),)),
        QuayCraneSpec("qc_feeder", 1.35, ("b_mid", "b_feeder"), (("reefer", 1.10),)),
    )
    yards = (
        YardBlockSpec("y_near", 0.0, ("general", "reefer"), (("general", 2.60), ("reefer", 2.20))),
        YardBlockSpec("y_bulk", 1.2, ("general", "bulk"), (("general", 1.90), ("bulk", 2.75))),
        YardBlockSpec("y_far", 2.5, ("general", "reefer", "bulk"), (("general", 1.45), ("reefer", 1.25), ("bulk", 1.65))),
    )
    gates = (
        GateLaneSpec("g_priority", ("general", "reefer"), (("general", 2.35), ("reefer", 2.00))),
        GateLaneSpec("g_freight", ("general", "bulk"), (("general", 1.70), ("bulk", 2.30))),
    )
    return {"berths": berths, "quay_cranes": cranes, "yard_blocks": yards, "gate_lanes": gates}


def _vessel(
    vessel_id: str,
    arrival: float,
    due: float,
    cargo: str,
    work: tuple[float, float, float],
    size: tuple[float, float],
    preferred: tuple[str, str],
    *,
    priority: float = 1.0,
    max_cranes: int = 2,
) -> VesselSpec:
    return VesselSpec(
        vessel_id=vessel_id,
        arrival_time=arrival,
        due_time=due,
        priority_weight=priority,
        length=size[0],
        draft=size[1],
        cargo_class=cargo,
        quay_work=work[0],
        yard_work=work[1],
        gate_work=work[2],
        preferred_berth=preferred[0],
        preferred_yard=preferred[1],
        max_cranes=max_cranes,
    )


def _balanced_instance() -> PortInstance:
    resources = _resource_bundle()
    return PortInstance(
        name="balanced_heterogeneous_8",
        **resources,
        vessels=(
            _vessel("v01", 0, 165, "general", (150, 75, 48), (260, 11.5), ("b_mid", "y_near"), priority=1.2),
            _vessel("v02", 3, 92, "reefer", (112, 68, 44), (205, 9.5), ("b_feeder", "y_near"), priority=1.5),
            _vessel("v03", 8, 150, "bulk", (185, 92, 58), (330, 14.0), ("b_deep", "y_bulk"), max_cranes=3),
            _vessel("v04", 14, 88, "general", (96, 60, 39), (195, 9.0), ("b_feeder", "y_far")),
            _vessel("v05", 21, 118, "reefer", (138, 84, 52), (275, 12.0), ("b_mid", "y_near"), priority=1.3),
            _vessel("v06", 29, 180, "bulk", (125, 72, 54), (215, 9.8), ("b_feeder", "y_bulk")),
            _vessel("v07", 37, 135, "general", (172, 88, 62), (345, 14.5), ("b_deep", "y_near"), priority=1.4, max_cranes=3),
            _vessel("v08", 48, 108, "general", (105, 57, 41), (230, 10.5), ("b_mid", "y_far")),
        ),
    )


def _berth_crane_pressure_instance() -> PortInstance:
    resources = _resource_bundle()
    return PortInstance(
        name="berth_crane_pressure_9",
        **resources,
        decision_interval=6.0,
        vessels=(
            _vessel("p01", 0, 165, "general", (205, 58, 36), (350, 14.7), ("b_deep", "y_near"), priority=1.8, max_cranes=3),
            _vessel("p02", 1, 74, "reefer", (82, 48, 31), (190, 8.8), ("b_feeder", "y_near"), priority=1.7),
            _vessel("p03", 2, 145, "bulk", (178, 68, 45), (280, 12.0), ("b_mid", "y_bulk"), max_cranes=3),
            _vessel("p04", 4, 82, "general", (104, 52, 35), (215, 9.8), ("b_feeder", "y_far"), priority=1.3),
            _vessel("p05", 10, 108, "general", (165, 66, 42), (325, 13.8), ("b_deep", "y_near"), max_cranes=3),
            _vessel("p06", 18, 96, "reefer", (121, 72, 44), (268, 11.8), ("b_mid", "y_near"), priority=1.4),
            _vessel("p07", 24, 154, "bulk", (136, 74, 49), (210, 9.5), ("b_feeder", "y_bulk")),
            _vessel("p08", 31, 126, "general", (148, 78, 51), (300, 12.3), ("b_deep", "y_far"), priority=1.2),
            _vessel("p09", 43, 92, "reefer", (92, 56, 38), (200, 9.1), ("b_feeder", "y_near")),
        ),
    )


def _yard_gate_surge_instance() -> PortInstance:
    resources = _resource_bundle()
    return PortInstance(
        name="yard_gate_surge_9",
        **resources,
        decision_interval=7.0,
        yard_rehandle_base_duration=3.5,
        vessels=(
            _vessel("s01", 0, 170, "general", (92, 120, 72), (230, 10.0), ("b_mid", "y_near"), priority=1.3),
            _vessel("s02", 2, 120, "bulk", (102, 135, 78), (205, 9.2), ("b_feeder", "y_bulk"), priority=1.4),
            _vessel("s03", 4, 95, "reefer", (88, 128, 70), (198, 8.9), ("b_feeder", "y_near"), priority=1.6),
            _vessel("s04", 11, 145, "general", (115, 142, 86), (285, 12.2), ("b_mid", "y_far")),
            _vessel("s05", 18, 112, "bulk", (125, 155, 91), (340, 14.0), ("b_deep", "y_bulk"), max_cranes=3),
            _vessel("s06", 25, 105, "reefer", (94, 118, 74), (215, 9.8), ("b_feeder", "y_far"), priority=1.5),
            _vessel("s07", 33, 158, "general", (108, 150, 89), (270, 11.7), ("b_mid", "y_near")),
            _vessel("s08", 41, 132, "bulk", (132, 162, 95), (315, 13.0), ("b_deep", "y_bulk"), priority=1.2),
            _vessel("s09", 52, 104, "general", (86, 106, 68), (195, 9.0), ("b_feeder", "y_far")),
        ),
    )


def _unique_ids(label: str, values: Iterable[str]) -> set[str]:
    rows = [str(value) for value in values]
    if any(not value for value in rows) or len(rows) != len(set(rows)):
        raise ValueError(f"{label} identifiers must be non-empty and unique")
    return set(rows)


def _require_positive(value: float, label: str) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{label} must be positive")


__all__ = [
    "BerthSpec",
    "GateLaneSpec",
    "PortInstance",
    "QuayCraneSpec",
    "VesselSpec",
    "YardBlockSpec",
    "built_in_port_instances",
    "load_port_instance",
    "port_instance_from_mapping",
]
