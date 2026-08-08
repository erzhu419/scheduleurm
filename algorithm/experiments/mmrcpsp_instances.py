"""Strict core parser and feasible-action model for PSPLIB ``.mm`` files.

The parser intentionally covers the single-project, multi-mode PSPLIB format
with renewable and nonrenewable resources.  Unsupported extensions fail closed
instead of being silently interpreted as the core MMRCPSP model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable, Sequence


_INTEGER_RE = re.compile(r"-?\d+")


class MMRCPSPFormatError(ValueError):
    """Raised when a file is outside the supported PSPLIB ``.mm`` contract."""


@dataclass(frozen=True)
class Mode:
    job_id: int
    mode_id: int
    duration: int
    renewable_demands: tuple[int, ...]
    nonrenewable_demands: tuple[int, ...]

    def snapshot(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "mode_id": self.mode_id,
            "duration": self.duration,
            "renewable_demands": list(self.renewable_demands),
            "nonrenewable_demands": list(self.nonrenewable_demands),
        }


@dataclass(frozen=True)
class Activity:
    job_id: int
    declared_mode_count: int
    successors: tuple[int, ...]
    predecessors: tuple[int, ...]
    modes: tuple[Mode, ...]

    def snapshot(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "declared_mode_count": self.declared_mode_count,
            "successors": list(self.successors),
            "predecessors": list(self.predecessors),
            "modes": [mode.snapshot() for mode in self.modes],
        }


@dataclass(frozen=True)
class ModeAction:
    """A mode choice that is feasible in the supplied scheduling state."""

    job_id: int
    mode_id: int
    duration: int
    renewable_demands: tuple[int, ...]
    nonrenewable_demands: tuple[int, ...]
    nonrenewable_reserve_after: tuple[int, ...]

    @property
    def action_id(self) -> str:
        return f"job_{self.job_id}:mode_{self.mode_id}"

    def snapshot(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "job_id": self.job_id,
            "mode_id": self.mode_id,
            "duration": self.duration,
            "renewable_demands": list(self.renewable_demands),
            "nonrenewable_demands": list(self.nonrenewable_demands),
            "nonrenewable_reserve_after": list(
                self.nonrenewable_reserve_after
            ),
        }


@dataclass(frozen=True)
class MMRCPSPInstance:
    name: str
    source_name: str
    source_sha256: str
    project_count: int
    declared_job_count: int
    horizon: int
    renewable_capacities: tuple[int, ...]
    nonrenewable_capacities: tuple[int, ...]
    activities: tuple[Activity, ...]

    @property
    def activity_map(self) -> dict[int, Activity]:
        return {activity.job_id: activity for activity in self.activities}

    @property
    def job_ids(self) -> tuple[int, ...]:
        return tuple(activity.job_id for activity in self.activities)

    @property
    def real_job_ids(self) -> tuple[int, ...]:
        """Return non-dummy jobs used by completion-time metrics."""

        return tuple(
            activity.job_id
            for activity in self.activities
            if any(
                mode.duration > 0
                or any(mode.renewable_demands)
                or any(mode.nonrenewable_demands)
                for mode in activity.modes
            )
        )

    @property
    def source_job_ids(self) -> tuple[int, ...]:
        return tuple(
            activity.job_id
            for activity in self.activities
            if not activity.predecessors
        )

    @property
    def sink_job_ids(self) -> tuple[int, ...]:
        return tuple(
            activity.job_id
            for activity in self.activities
            if not activity.successors
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "project_count": self.project_count,
            "declared_job_count": self.declared_job_count,
            "horizon": self.horizon,
            "renewable_capacities": list(self.renewable_capacities),
            "nonrenewable_capacities": list(self.nonrenewable_capacities),
            "source_job_ids": list(self.source_job_ids),
            "sink_job_ids": list(self.sink_job_ids),
            "real_job_ids": list(self.real_job_ids),
            "activities": [activity.snapshot() for activity in self.activities],
        }


def parse_psplib_mm(path: str | Path) -> MMRCPSPInstance:
    """Parse a single-project PSPLIB multi-mode instance from ``path``."""

    source = Path(path)
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MMRCPSPFormatError(
            f"{source.name}: PSPLIB text must be UTF-8/ASCII"
        ) from exc
    return parse_psplib_mm_text(
        text,
        name=source.stem,
        source_name=source.name,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def parse_psplib_mm_text(
    text: str,
    *,
    name: str = "in_memory",
    source_name: str = "in_memory.mm",
    source_sha256: str | None = None,
) -> MMRCPSPInstance:
    """Parse the supported core of a PSPLIB ``.mm`` document."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    digest = source_sha256 or hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    lines = normalized.splitlines()

    project_count = _required_scalar(normalized, r"(?im)^\s*projects\s*:\s*(\d+)", "projects")
    if project_count != 1:
        raise MMRCPSPFormatError(
            f"{source_name}: only single-project PSPLIB .mm files are supported"
        )
    declared_job_count = _required_scalar(
        normalized,
        r"(?im)^\s*jobs\s*\(incl\.\s*supersource/sink\s*\)\s*:\s*(\d+)",
        "jobs (incl. supersource/sink)",
    )
    horizon = _required_scalar(normalized, r"(?im)^\s*horizon\s*:\s*(\d+)", "horizon")
    renewable_count = _required_scalar(
        normalized,
        r"(?im)^\s*-\s*renewable\s*:\s*(\d+)",
        "renewable resource count",
    )
    nonrenewable_count = _required_scalar(
        normalized,
        r"(?im)^\s*-\s*nonrenewable\s*:\s*(\d+)",
        "nonrenewable resource count",
    )
    doubly_constrained_count = _required_scalar(
        normalized,
        r"(?im)^\s*-\s*doubly\s+constrained\s*:\s*(\d+)",
        "doubly constrained resource count",
    )
    if doubly_constrained_count:
        raise MMRCPSPFormatError(
            f"{source_name}: doubly constrained resources are outside the core R/N parser"
        )

    precedence_index = _section_index(lines, "PRECEDENCE RELATIONS:", source_name)
    requests_index = _section_index(lines, "REQUESTS/DURATIONS:", source_name)
    capacities_index = _section_index(lines, "RESOURCEAVAILABILITIES:", source_name)
    if not precedence_index < requests_index < capacities_index:
        raise MMRCPSPFormatError(f"{source_name}: PSPLIB sections are out of order")

    declarations = _parse_precedence(
        lines[precedence_index + 1 : requests_index], source_name
    )
    modes = _parse_modes(
        lines[requests_index + 1 : capacities_index],
        renewable_count=renewable_count,
        nonrenewable_count=nonrenewable_count,
        source_name=source_name,
    )
    capacities = _parse_capacities(
        lines[capacities_index + 1 :],
        resource_count=renewable_count + nonrenewable_count,
        source_name=source_name,
    )

    if len(declarations) != declared_job_count:
        raise MMRCPSPFormatError(
            f"{source_name}: declared {declared_job_count} jobs but parsed {len(declarations)}"
        )
    declared_jobs = set(declarations)
    if set(modes) != declared_jobs:
        missing = sorted(declared_jobs - set(modes))
        extra = sorted(set(modes) - declared_jobs)
        raise MMRCPSPFormatError(
            f"{source_name}: mode rows do not match jobs; missing={missing}, extra={extra}"
        )

    predecessors: dict[int, list[int]] = {job_id: [] for job_id in declarations}
    for job_id, (_, successors) in declarations.items():
        for successor in successors:
            if successor not in declarations:
                raise MMRCPSPFormatError(
                    f"{source_name}: job {job_id} references unknown successor {successor}"
                )
            predecessors[successor].append(job_id)

    activities: list[Activity] = []
    for job_id in sorted(declarations):
        declared_modes, successors = declarations[job_id]
        parsed_modes = tuple(sorted(modes[job_id], key=lambda mode: mode.mode_id))
        if len(parsed_modes) != declared_modes:
            raise MMRCPSPFormatError(
                f"{source_name}: job {job_id} declares {declared_modes} modes "
                f"but has {len(parsed_modes)} request rows"
            )
        if len({mode.mode_id for mode in parsed_modes}) != len(parsed_modes):
            raise MMRCPSPFormatError(
                f"{source_name}: duplicate mode id for job {job_id}"
            )
        expected_mode_ids = set(range(1, declared_modes + 1))
        if {mode.mode_id for mode in parsed_modes} != expected_mode_ids:
            raise MMRCPSPFormatError(
                f"{source_name}: job {job_id} mode ids must be contiguous 1..{declared_modes}"
            )
        activities.append(
            Activity(
                job_id=job_id,
                declared_mode_count=declared_modes,
                successors=successors,
                predecessors=tuple(sorted(predecessors[job_id])),
                modes=parsed_modes,
            )
        )

    _validate_acyclic(tuple(activities), source_name)
    _validate_supersource_and_sink(tuple(activities), source_name)
    renewable_capacities = tuple(capacities[:renewable_count])
    nonrenewable_capacities = tuple(capacities[renewable_count:])
    return MMRCPSPInstance(
        name=name,
        source_name=source_name,
        source_sha256=digest,
        project_count=project_count,
        declared_job_count=declared_job_count,
        horizon=horizon,
        renewable_capacities=renewable_capacities,
        nonrenewable_capacities=nonrenewable_capacities,
        activities=tuple(activities),
    )


def feasible_mode_actions(
    instance: MMRCPSPInstance,
    *,
    eligible_job_ids: Iterable[int],
    unscheduled_job_ids: Iterable[int],
    renewable_available: Sequence[int],
    nonrenewable_consumed: Sequence[int],
) -> tuple[ModeAction, ...]:
    """Enumerate state-feasible modes with a nonrenewable reserve certificate.

    For every candidate, an exact finite Pareto-frontier dynamic program checks
    whether one mode can still be selected for every remaining activity within
    all nonrenewable budgets simultaneously.  This avoids the false certificate
    obtained by minimizing each resource coordinate with potentially different
    modes.
    """

    renewable_available_tuple = tuple(int(value) for value in renewable_available)
    nonrenewable_consumed_tuple = tuple(
        int(value) for value in nonrenewable_consumed
    )
    if len(renewable_available_tuple) != len(instance.renewable_capacities):
        raise ValueError("renewable availability dimension mismatch")
    if len(nonrenewable_consumed_tuple) != len(
        instance.nonrenewable_capacities
    ):
        raise ValueError("nonrenewable consumption dimension mismatch")

    activity_map = instance.activity_map
    eligible = sorted(set(int(job_id) for job_id in eligible_job_ids))
    unscheduled = set(int(job_id) for job_id in unscheduled_job_ids)
    unknown = (set(eligible) | unscheduled) - set(activity_map)
    if unknown:
        raise ValueError(f"unknown job ids in scheduling state: {sorted(unknown)}")

    actions: list[ModeAction] = []
    for job_id in eligible:
        if job_id not in unscheduled:
            continue
        for mode in activity_map[job_id].modes:
            if any(
                demand > available
                for demand, available in zip(
                    mode.renewable_demands,
                    renewable_available_tuple,
                    strict=True,
                )
            ):
                continue
            residual_capacity = tuple(
                capacity - consumed - demand
                for capacity, consumed, demand in zip(
                    instance.nonrenewable_capacities,
                    nonrenewable_consumed_tuple,
                    mode.nonrenewable_demands,
                    strict=True,
                )
            )
            if any(value < 0 for value in residual_capacity):
                continue
            completion_usage = _minimum_feasible_nonrenewable_completion(
                instance,
                remaining_job_ids=sorted(unscheduled - {job_id}),
                residual_capacity=residual_capacity,
            )
            if completion_usage is None:
                continue
            reserve_after = tuple(
                consumed + demand + future
                for consumed, demand, future in zip(
                    nonrenewable_consumed_tuple,
                    mode.nonrenewable_demands,
                    completion_usage,
                    strict=True,
                )
            )
            actions.append(
                ModeAction(
                    job_id=job_id,
                    mode_id=mode.mode_id,
                    duration=mode.duration,
                    renewable_demands=mode.renewable_demands,
                    nonrenewable_demands=mode.nonrenewable_demands,
                    nonrenewable_reserve_after=reserve_after,
                )
            )
    return tuple(sorted(actions, key=lambda row: (row.job_id, row.mode_id)))


def _required_scalar(text: str, pattern: str, label: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise MMRCPSPFormatError(f"missing PSPLIB field: {label}")
    return int(match.group(1))


def _section_index(lines: Sequence[str], heading: str, source_name: str) -> int:
    for index, line in enumerate(lines):
        if line.strip().upper() == heading:
            return index
    raise MMRCPSPFormatError(f"{source_name}: missing section {heading}")


def _parse_precedence(
    lines: Sequence[str], source_name: str
) -> dict[int, tuple[int, tuple[int, ...]]]:
    declarations: dict[int, tuple[int, tuple[int, ...]]] = {}
    for line in lines:
        if not re.match(r"^\s*\d", line):
            continue
        values = [int(value) for value in _INTEGER_RE.findall(line)]
        if len(values) < 3:
            raise MMRCPSPFormatError(
                f"{source_name}: malformed precedence row: {line.strip()}"
            )
        job_id, mode_count, successor_count = values[:3]
        successors = tuple(values[3:])
        if mode_count <= 0 or successor_count < 0:
            raise MMRCPSPFormatError(
                f"{source_name}: invalid precedence counts for job {job_id}"
            )
        if len(successors) != successor_count:
            raise MMRCPSPFormatError(
                f"{source_name}: job {job_id} declares {successor_count} successors "
                f"but lists {len(successors)}"
            )
        if job_id in declarations:
            raise MMRCPSPFormatError(
                f"{source_name}: duplicate precedence row for job {job_id}"
            )
        declarations[job_id] = (mode_count, successors)
    if not declarations:
        raise MMRCPSPFormatError(f"{source_name}: no precedence rows found")
    return declarations


def _parse_modes(
    lines: Sequence[str],
    *,
    renewable_count: int,
    nonrenewable_count: int,
    source_name: str,
) -> dict[int, list[Mode]]:
    resource_count = renewable_count + nonrenewable_count
    modes: dict[int, list[Mode]] = {}
    current_job: int | None = None
    for line in lines:
        if not re.match(r"^\s*\d", line):
            continue
        values = [int(value) for value in _INTEGER_RE.findall(line)]
        if len(values) == resource_count + 3:
            job_id, mode_id, duration = values[:3]
            demands = values[3:]
            current_job = job_id
        elif len(values) == resource_count + 2 and current_job is not None:
            job_id = current_job
            mode_id, duration = values[:2]
            demands = values[2:]
        else:
            raise MMRCPSPFormatError(
                f"{source_name}: malformed request/duration row: {line.strip()}"
            )
        if mode_id <= 0 or duration < 0 or any(value < 0 for value in demands):
            raise MMRCPSPFormatError(
                f"{source_name}: negative duration/demand or invalid mode for job {job_id}"
            )
        modes.setdefault(job_id, []).append(
            Mode(
                job_id=job_id,
                mode_id=mode_id,
                duration=duration,
                renewable_demands=tuple(demands[:renewable_count]),
                nonrenewable_demands=tuple(demands[renewable_count:]),
            )
        )
    if not modes:
        raise MMRCPSPFormatError(f"{source_name}: no request/duration rows found")
    return modes


def _parse_capacities(
    lines: Sequence[str], *, resource_count: int, source_name: str
) -> tuple[int, ...]:
    if resource_count == 0:
        return ()
    for line in lines:
        if not re.match(r"^\s*\d", line):
            continue
        values = [int(value) for value in _INTEGER_RE.findall(line)]
        if len(values) != resource_count:
            raise MMRCPSPFormatError(
                f"{source_name}: expected {resource_count} resource capacities, "
                f"found {len(values)}"
            )
        if any(value < 0 for value in values):
            raise MMRCPSPFormatError(
                f"{source_name}: resource capacities must be nonnegative"
            )
        return tuple(values)
    raise MMRCPSPFormatError(f"{source_name}: no resource capacities found")


def _validate_acyclic(
    activities: Sequence[Activity], source_name: str
) -> None:
    indegree = {
        activity.job_id: len(activity.predecessors) for activity in activities
    }
    successors = {
        activity.job_id: activity.successors for activity in activities
    }
    ready = sorted(job_id for job_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        job_id = ready.pop(0)
        visited += 1
        for successor in successors[job_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if visited != len(activities):
        raise MMRCPSPFormatError(
            f"{source_name}: precedence graph contains a directed cycle"
        )


def _validate_supersource_and_sink(
    activities: Sequence[Activity], source_name: str
) -> None:
    sources = [activity for activity in activities if not activity.predecessors]
    sinks = [activity for activity in activities if not activity.successors]
    if len(sources) != 1 or len(sinks) != 1:
        raise MMRCPSPFormatError(
            f"{source_name}: core PSPLIB project requires one supersource and one supersink"
        )
    for role, activity in (("supersource", sources[0]), ("supersink", sinks[0])):
        if len(activity.modes) != 1:
            raise MMRCPSPFormatError(
                f"{source_name}: {role} must declare exactly one mode"
            )
        mode = activity.modes[0]
        if (
            mode.duration != 0
            or any(mode.renewable_demands)
            or any(mode.nonrenewable_demands)
        ):
            raise MMRCPSPFormatError(
                f"{source_name}: {role} must have zero duration and zero demand"
            )


def _minimum_feasible_nonrenewable_completion(
    instance: MMRCPSPInstance,
    *,
    remaining_job_ids: Sequence[int],
    residual_capacity: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Return one minimal feasible residual-use vector, or ``None``.

    Dominated cumulative-use vectors can be removed exactly because all demands
    are nonnegative and the only nonrenewable constraint is an upper budget.
    """

    width = len(instance.nonrenewable_capacities)
    if width == 0:
        return ()
    activity_map = instance.activity_map
    frontier: set[tuple[int, ...]] = {tuple(0 for _ in range(width))}
    for job_id in sorted(remaining_job_ids):
        next_states: set[tuple[int, ...]] = set()
        for used in frontier:
            for mode in activity_map[job_id].modes:
                candidate = tuple(
                    old + demand
                    for old, demand in zip(
                        used, mode.nonrenewable_demands, strict=True
                    )
                )
                if all(
                    value <= capacity
                    for value, capacity in zip(
                        candidate, residual_capacity, strict=True
                    )
                ):
                    next_states.add(candidate)
        if not next_states:
            return None
        frontier = _pareto_minimal_vectors(next_states)
    return min(frontier, key=lambda row: (sum(row), row))


def _pareto_minimal_vectors(
    vectors: Iterable[tuple[int, ...]],
) -> set[tuple[int, ...]]:
    ordered = sorted(set(vectors), key=lambda row: (sum(row), row))
    frontier: list[tuple[int, ...]] = []
    for candidate in ordered:
        if any(
            all(old <= new for old, new in zip(existing, candidate, strict=True))
            for existing in frontier
        ):
            continue
        frontier = [
            existing
            for existing in frontier
            if not all(
                new <= old
                for new, old in zip(candidate, existing, strict=True)
            )
        ]
        frontier.append(candidate)
    return set(frontier)
